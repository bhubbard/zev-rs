use serde_json::{json, Value};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use zev::mcp::{handle_message, process_message, run_mcp_server_io};
use zev::ZevEngine;

#[tokio::test]
async fn test_mcp_initialize_handshake() {
    let engine = ZevEngine::default();
    let req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    });

    let resp_str = process_message(&req.to_string(), &engine)
        .await
        .expect("Expected response for initialize request");
    let resp: Value = serde_json::from_str(&resp_str).unwrap();

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    assert_eq!(resp["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(resp["result"]["serverInfo"]["name"], "zev-mcp");
    assert_eq!(resp["result"]["serverInfo"]["version"], env!("CARGO_PKG_VERSION"));
    assert!(resp["result"]["capabilities"]["tools"].is_object());
}

#[tokio::test]
async fn test_mcp_ping() {
    let engine = ZevEngine::default();
    let req = json!({
        "jsonrpc": "2.0",
        "id": "ping-42",
        "method": "ping"
    });

    let resp_str = process_message(&req.to_string(), &engine)
        .await
        .expect("Expected response for ping");
    let resp: Value = serde_json::from_str(&resp_str).unwrap();

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], "ping-42");
    assert_eq!(resp["result"], json!({}));
}

#[tokio::test]
async fn test_mcp_notifications_produce_no_response() {
    let engine = ZevEngine::default();

    // 1. notifications/initialized without id
    let req1 = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });
    assert!(process_message(&req1.to_string(), &engine).await.is_none());

    // 2. notifications/initialized with id (must still be suppressed)
    let req2 = json!({
        "jsonrpc": "2.0",
        "id": 99,
        "method": "notifications/initialized"
    });
    assert!(process_message(&req2.to_string(), &engine).await.is_none());

    // 3. Generic notification without id
    let req3 = json!({
        "jsonrpc": "2.0",
        "method": "some/notification"
    });
    assert!(process_message(&req3.to_string(), &engine).await.is_none());
}

#[tokio::test]
async fn test_mcp_tools_list_schema() {
    let engine = ZevEngine::default();
    let req = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    });

    let resp_str = process_message(&req.to_string(), &engine)
        .await
        .expect("Expected response for tools/list");
    let resp: Value = serde_json::from_str(&resp_str).unwrap();

    let tools = resp["result"]["tools"].as_array().expect("tools array");
    assert_eq!(tools.len(), 4);

    let tool_names: Vec<&str> = tools
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();

    assert!(tool_names.contains(&"zev_classify"));
    assert!(tool_names.contains(&"zev_filter"));
    assert!(tool_names.contains(&"zev_score"));
    assert!(tool_names.contains(&"zev_batch"));

    // Verify zev_classify schema
    let classify = tools.iter().find(|t| t["name"] == "zev_classify").unwrap();
    assert!(classify["inputSchema"]["properties"]["state"].is_object());
    assert!(classify["inputSchema"]["properties"]["instructions"].is_object());
    assert!(classify["inputSchema"]["properties"]["options"].is_object());
    let classify_req = classify["inputSchema"]["required"].as_array().unwrap();
    assert!(classify_req.iter().any(|v| v == "state"));
    assert!(classify_req.iter().any(|v| v == "instructions"));
    assert!(classify_req.iter().any(|v| v == "options"));

    // Verify zev_filter schema
    let filter = tools.iter().find(|t| t["name"] == "zev_filter").unwrap();
    assert!(filter["inputSchema"]["properties"]["state"].is_object());
    assert!(filter["inputSchema"]["properties"]["predicate"].is_object());
    assert!(filter["inputSchema"]["properties"]["threshold"].is_object());
    let filter_req = filter["inputSchema"]["required"].as_array().unwrap();
    assert!(filter_req.iter().any(|v| v == "state"));
    assert!(filter_req.iter().any(|v| v == "predicate"));

    // Verify zev_score schema
    let score = tools.iter().find(|t| t["name"] == "zev_score").unwrap();
    assert!(score["inputSchema"]["properties"]["state"].is_object());
    assert!(score["inputSchema"]["properties"]["instructions"].is_object());
    assert!(score["inputSchema"]["properties"]["levels"].is_object());
    let score_req = score["inputSchema"]["required"].as_array().unwrap();
    assert!(score_req.iter().any(|v| v == "state"));
    assert!(score_req.iter().any(|v| v == "instructions"));
    assert!(score_req.iter().any(|v| v == "levels"));

    // Verify zev_batch schema
    let batch = tools.iter().find(|t| t["name"] == "zev_batch").unwrap();
    assert!(batch["inputSchema"]["properties"]["rows"].is_object());
    assert!(batch["inputSchema"]["properties"]["predicate"].is_object());
    assert!(batch["inputSchema"]["properties"]["threshold"].is_object());
    let batch_req = batch["inputSchema"]["required"].as_array().unwrap();
    assert!(batch_req.iter().any(|v| v == "rows"));
    assert!(batch_req.iter().any(|v| v == "predicate"));
    assert!(batch_req.iter().any(|v| v == "threshold"));
}

#[tokio::test]
async fn test_mcp_tools_call_classify() {
    let engine = ZevEngine::default();
    let req = json!({
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {
            "name": "zev_classify",
            "arguments": {
                "state": "Customer calls about credit card charge dispute and wants a refund on the billing invoice.",
                "instructions": "Classify customer support inquiry destination",
                "options": [
                    { "id": "billing", "description": "Payment processing, invoices, and charge refunds" },
                    { "id": "technical", "description": "Server crashes, API bugs, and database outages" },
                    { "id": "sales", "description": "Enterprise software licensing and new subscriptions" }
                ]
            }
        }
    });

    let resp_str = process_message(&req.to_string(), &engine).await.unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();

    assert_eq!(resp["id"], 10);
    assert_eq!(resp["result"]["isError"], false);

    let content_text = resp["result"]["content"][0]["text"].as_str().unwrap();
    let res: Value = serde_json::from_str(content_text).unwrap();

    assert_eq!(res["winning_option"], "billing");
    assert_eq!(res["decision"], "billing");
    assert!(res["confidence"].as_f64().unwrap() > 0.5);
    assert!(res["probabilities"]["billing"].as_f64().unwrap() > res["probabilities"]["technical"].as_f64().unwrap());
    assert!(res["probabilities"]["billing"].as_f64().unwrap() > res["probabilities"]["sales"].as_f64().unwrap());
}

#[tokio::test]
async fn test_mcp_tools_call_filter() {
    let engine = ZevEngine::default();

    // 1. Positive case: should pass filter
    let req_pass = json!({
        "jsonrpc": "2.0",
        "id": 11,
        "method": "tools/call",
        "params": {
            "name": "zev_filter",
            "arguments": {
                "state": "Urgent technical outage: production database crashed and connection refused 500",
                "predicate": "urgent technical outage or system failure",
                "threshold": 0.5
            }
        }
    });

    let resp_str = process_message(&req_pass.to_string(), &engine).await.unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["id"], 11);
    assert_eq!(resp["result"]["isError"], false);

    let content_text = resp["result"]["content"][0]["text"].as_str().unwrap();
    let res_pass: Value = serde_json::from_str(content_text).unwrap();
    assert_eq!(res_pass["pass"], true);
    assert!(res_pass["confidence"].as_f64().unwrap() > 0.5);

    // 2. Negative case: should fail filter
    let req_fail = json!({
        "jsonrpc": "2.0",
        "id": 12,
        "method": "tools/call",
        "params": {
            "name": "zev_filter",
            "arguments": {
                "state": "I really enjoy having chocolate ice cream on warm sunny afternoons.",
                "predicate": "urgent technical outage or system failure",
                "threshold": 0.5
            }
        }
    });

    let resp_str = process_message(&req_fail.to_string(), &engine).await.unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["id"], 12);
    assert_eq!(resp["result"]["isError"], false);

    let content_text = resp["result"]["content"][0]["text"].as_str().unwrap();
    let res_fail: Value = serde_json::from_str(content_text).unwrap();
    assert_eq!(res_fail["pass"], false);
}

#[tokio::test]
async fn test_mcp_tools_call_score() {
    let engine = ZevEngine::default();
    let req = json!({
        "jsonrpc": "2.0",
        "id": 13,
        "method": "tools/call",
        "params": {
            "name": "zev_score",
            "arguments": {
                "state": "Customer reported a terrible experience with poor quality and defective product.",
                "instructions": "Rate product quality and customer satisfaction",
                "levels": ["terrible", "poor", "acceptable", "good", "excellent"]
            }
        }
    });

    let resp_str = process_message(&req.to_string(), &engine).await.unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();

    assert_eq!(resp["id"], 13);
    assert_eq!(resp["result"]["isError"], false);

    let content_text = resp["result"]["content"][0]["text"].as_str().unwrap();
    let res: Value = serde_json::from_str(content_text).unwrap();

    // Catastrophic failure should score low on the scale (closer to terrible=0 / poor=1)
    let score = res["score"].as_f64().unwrap();
    let expected_val = res["expected_value"].as_f64().unwrap();
    assert_eq!(score, expected_val);
    assert!(score < 1.5, "Score should be low for terrible experience, got {}", score);

    // Verify probabilities are populated for all levels
    assert!(res["level_probabilities"]["terrible"].as_f64().unwrap() > 0.0);
    assert!(res["level_probabilities"]["excellent"].as_f64().unwrap() < res["level_probabilities"]["terrible"].as_f64().unwrap());
}

#[tokio::test]
async fn test_mcp_tools_call_batch() {
    let engine = ZevEngine::default();
    let req = json!({
        "jsonrpc": "2.0",
        "id": 14,
        "method": "tools/call",
        "params": {
            "name": "zev_batch",
            "arguments": {
                "rows": [
                    { "id": "row_1", "text": "Production database outage: connection refused 500 crash" },
                    { "id": "row_2", "text": "I love the new UI dashboard, great design!" },
                    { "id": "row_3", "text": "Critical API gateway outage: all endpoints returning 503 failure" },
                    { "id": "row_4", "text": "General question about office location" }
                ],
                "predicate": "urgent technical outage or system failure",
                "threshold": 0.5
            }
        }
    });

    let resp_str = process_message(&req.to_string(), &engine).await.unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();

    assert_eq!(resp["id"], 14);
    assert_eq!(resp["result"]["isError"], false);

    let content_text = resp["result"]["content"][0]["text"].as_str().unwrap();
    let res: Value = serde_json::from_str(content_text).unwrap();

    let rows = res["rows"].as_array().expect("rows array");
    assert_eq!(res["count"], 2);
    assert_eq!(rows.len(), 2);

    let row_ids: Vec<&str> = rows.iter().map(|r| r["id"].as_str().unwrap()).collect();
    assert!(row_ids.contains(&"row_1"));
    assert!(row_ids.contains(&"row_3"));
    assert!(!row_ids.contains(&"row_2"));
    assert!(!row_ids.contains(&"row_4"));
}

#[tokio::test]
async fn test_mcp_error_handling() {
    let engine = ZevEngine::default();

    // 1. JSON parse error
    let resp_str = handle_message("invalid json content", &engine).unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["error"]["code"], -32700);

    // 2. Method not found
    let req_unknown = json!({
        "jsonrpc": "2.0",
        "id": 90,
        "method": "unsupported/method"
    });
    let resp_str = process_message(&req_unknown.to_string(), &engine).await.unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["error"]["code"], -32601);

    // 3. Unknown tool
    let req_bad_tool = json!({
        "jsonrpc": "2.0",
        "id": 91,
        "method": "tools/call",
        "params": {
            "name": "nonexistent_tool",
            "arguments": {}
        }
    });
    let resp_str = process_message(&req_bad_tool.to_string(), &engine).await.unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["result"]["isError"], true);

    // 4. Missing required args in tool call
    let req_missing_args = json!({
        "jsonrpc": "2.0",
        "id": 92,
        "method": "tools/call",
        "params": {
            "name": "zev_classify",
            "arguments": {
                "state": "some text"
                // missing instructions and options
            }
        }
    });
    let resp_str = process_message(&req_missing_args.to_string(), &engine).await.unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["result"]["isError"], true);

    // 5. Insufficient options (< 2)
    let req_few_options = json!({
        "jsonrpc": "2.0",
        "id": 93,
        "method": "tools/call",
        "params": {
            "name": "zev_classify",
            "arguments": {
                "state": "some text",
                "instructions": "classify",
                "options": [{ "id": "only_one", "description": "desc" }]
            }
        }
    });
    let resp_str = process_message(&req_few_options.to_string(), &engine).await.unwrap();
    let resp: Value = serde_json::from_str(&resp_str).unwrap();
    assert_eq!(resp["result"]["isError"], true);
}

#[tokio::test]
async fn test_async_duplex_mcp_server() {
    let engine = Arc::new(ZevEngine::default());

    // Create duplex streams: (client_write, server_read) and (server_write, client_read)
    let (client_write, server_read) = tokio::io::duplex(4096);
    let (server_write, client_read) = tokio::io::duplex(4096);

    let server_handle = tokio::spawn(async move {
        run_mcp_server_io(engine, server_read, server_write)
            .await
            .unwrap();
    });

    let mut client_in = BufReader::new(client_read).lines();
    let mut client_out = client_write;

    // 1. Send initialize
    let init_req = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": { "name": "duplex-test", "version": "1.0" }
        }
    });
    client_out
        .write_all(format!("{}\n", init_req).as_bytes())
        .await
        .unwrap();
    client_out.flush().await.unwrap();

    let init_resp_line = client_in.next_line().await.unwrap().unwrap();
    let init_resp: Value = serde_json::from_str(&init_resp_line).unwrap();
    assert_eq!(init_resp["id"], 1);
    assert_eq!(init_resp["result"]["serverInfo"]["name"], "zev-mcp");

    // 2. Send tools/call for zev_filter
    let filter_req = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "zev_filter",
            "arguments": {
                "state": "API server is down with 500 error",
                "predicate": "outage or server error",
                "threshold": 0.5
            }
        }
    });
    client_out
        .write_all(format!("{}\n", filter_req).as_bytes())
        .await
        .unwrap();
    client_out.flush().await.unwrap();

    let filter_resp_line = client_in.next_line().await.unwrap().unwrap();
    let filter_resp: Value = serde_json::from_str(&filter_resp_line).unwrap();
    assert_eq!(filter_resp["id"], 2);
    assert_eq!(filter_resp["result"]["isError"], false);

    // 3. Send notification (must produce no response)
    let notif = json!({
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
    });
    client_out
        .write_all(format!("{}\n", notif).as_bytes())
        .await
        .unwrap();
    client_out.flush().await.unwrap();

    // 4. Send ping to ensure server is still responsive and didn't reply to notification
    let ping_req = json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "ping"
    });
    client_out
        .write_all(format!("{}\n", ping_req).as_bytes())
        .await
        .unwrap();
    client_out.flush().await.unwrap();

    let ping_resp_line = client_in.next_line().await.unwrap().unwrap();
    let ping_resp: Value = serde_json::from_str(&ping_resp_line).unwrap();
    assert_eq!(ping_resp["id"], 3);

    // 5. Close client stream to terminate server loop cleanly
    drop(client_out);
    server_handle.await.unwrap();
}

#[tokio::test]
async fn test_mcp_more_error_and_edge_cases() {
    let engine = ZevEngine::default();

    // 1. tools/call with missing tool name
    let req_missing_name = json!({
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {
            "arguments": {}
        }
    });
    let resp: Value = serde_json::from_str(&process_message(&req_missing_name.to_string(), &engine).await.unwrap()).unwrap();
    assert_eq!(resp["error"]["code"], -32602);
    assert!(resp["error"]["message"].as_str().unwrap().contains("Missing 'name'"));

    // 2. tools/call with missing params entirely
    let req_missing_params = json!({
        "jsonrpc": "2.0",
        "id": 102,
        "method": "tools/call"
    });
    let resp2: Value = serde_json::from_str(&process_message(&req_missing_params.to_string(), &engine).await.unwrap()).unwrap();
    assert_eq!(resp2["error"]["code"], -32602);
    assert!(resp2["error"]["message"].as_str().unwrap().contains("Missing params"));

    // 3. zev_classify with missing options field
    let req_classify_no_opts = json!({
        "jsonrpc": "2.0",
        "id": 103,
        "method": "tools/call",
        "params": {
            "name": "zev_classify",
            "arguments": {
                "state": "test",
                "instructions": "classify"
            }
        }
    });
    let resp3: Value = serde_json::from_str(&process_message(&req_classify_no_opts.to_string(), &engine).await.unwrap()).unwrap();
    assert_eq!(resp3["result"]["isError"], true);
    assert!(resp3["result"]["content"][0]["text"].as_str().unwrap().contains("Invalid arguments for zev_classify"));

    // 4. zev_score with fewer than 2 levels
    let req_score_short = json!({
        "jsonrpc": "2.0",
        "id": 104,
        "method": "tools/call",
        "params": {
            "name": "zev_score",
            "arguments": {
                "state": "latency is 200ms",
                "instructions": "rate latency",
                "levels": ["level 1"]
            }
        }
    });
    let resp4: Value = serde_json::from_str(&process_message(&req_score_short.to_string(), &engine).await.unwrap()).unwrap();
    assert_eq!(resp4["result"]["isError"], true);
    assert!(resp4["result"]["content"][0]["text"].as_str().unwrap().contains("at least 2 entries"));

    // 5. zev_score with invalid arguments
    let req_score_invalid = json!({
        "jsonrpc": "2.0",
        "id": 105,
        "method": "tools/call",
        "params": {
            "name": "zev_score",
            "arguments": {
                "state": 12345
            }
        }
    });
    let resp5: Value = serde_json::from_str(&process_message(&req_score_invalid.to_string(), &engine).await.unwrap()).unwrap();
    assert_eq!(resp5["result"]["isError"], true);
    assert!(resp5["result"]["content"][0]["text"].as_str().unwrap().contains("Invalid arguments for zev_score"));

    // 6. zev_filter with invalid arguments
    let req_filter_invalid = json!({
        "jsonrpc": "2.0",
        "id": 106,
        "method": "tools/call",
        "params": {
            "name": "zev_filter",
            "arguments": {
                "threshold": "not-a-float"
            }
        }
    });
    let resp6: Value = serde_json::from_str(&process_message(&req_filter_invalid.to_string(), &engine).await.unwrap()).unwrap();
    assert_eq!(resp6["result"]["isError"], true);
    assert!(resp6["result"]["content"][0]["text"].as_str().unwrap().contains("Invalid arguments for zev_filter"));

    // 7. zev_batch with invalid arguments
    let req_batch_invalid = json!({
        "jsonrpc": "2.0",
        "id": 107,
        "method": "tools/call",
        "params": {
            "name": "zev_batch",
            "arguments": {
                "rows": "not-an-array"
            }
        }
    });
    let resp7: Value = serde_json::from_str(&process_message(&req_batch_invalid.to_string(), &engine).await.unwrap()).unwrap();
    assert_eq!(resp7["result"]["isError"], true);
    assert!(resp7["result"]["content"][0]["text"].as_str().unwrap().contains("Invalid arguments for zev_batch"));

    // 8. zev_batch with empty rows
    let req_batch_empty = json!({
        "jsonrpc": "2.0",
        "id": 108,
        "method": "tools/call",
        "params": {
            "name": "zev_batch",
            "arguments": {
                "rows": [],
                "predicate": "anything",
                "threshold": 0.5
            }
        }
    });
    let resp8: Value = serde_json::from_str(&process_message(&req_batch_empty.to_string(), &engine).await.unwrap()).unwrap();
    assert_eq!(resp8["result"]["isError"], false);
    assert!(resp8["result"]["content"][0]["text"].as_str().unwrap().contains("\"count\": 0"));

    // 9. handle_message synchronous helper directly
    let sync_resp = handle_message("not valid json at all", &engine);
    assert!(sync_resp.is_some());
    let err_val: Value = serde_json::from_str(&sync_resp.unwrap()).unwrap();
    assert_eq!(err_val["error"]["code"], -32700);

    // 10. Empty JSON object (missing method)
    let sync_invalid = handle_message("{}", &engine);
    assert!(sync_invalid.is_some());
    let err_val2: Value = serde_json::from_str(&sync_invalid.unwrap()).unwrap();
    assert_eq!(err_val2["error"]["code"], -32600);
}
