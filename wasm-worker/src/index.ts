import { initSync, zev_evaluate_json, zev_evaluate_system_one_json } from "../pkg/zev.js";
import wasmModule from "../pkg/zev_bg.wasm";

// Initialize WASM module synchronously on worker startup
let initialized = false;
function ensureInit() {
  if (!initialized) {
    initSync(wasmModule);
    initialized = true;
  }
}

export default {
  async fetch(request: Request, env: any, ctx: ExecutionContext): Promise<Response> {
    ensureInit();

    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method.toUpperCase();

    // CORS headers
    const corsHeaders = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    };

    if (method === "OPTIONS") {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      // 1. Health & Info
      if (path === "/" && method === "GET") {
        return Response.json(
          {
            name: "zev-decision-worker",
            description: "Zero-Token LLM Decision Engine running on Cloudflare Workers WASM",
            engine: "zev-wasm-edge",
            version: "0.3.11",
            status: "ready",
            architecture: "Non-autoregressive order-invariant inverted index",
            latency_tier: "microsecond (<100µs in-isolate execution)",
            endpoints: {
              "POST /evaluate": "Evaluate full ZevRequest JSON payload",
              "POST /v1/system_one": "TypeSafe / Jev / Clef wire format compatible",
              "GET /bench": "Run in-isolate microbenchmark (500 decisions) with latency & throughput stats",
            },
          },
          { headers: corsHeaders }
        );
      }

      // 2. Microbenchmark inside V8 isolate
      if (path === "/bench" && method === "GET") {
        const sampleRequest = JSON.stringify({
          state: "Customer states transaction TX-9921 was debited twice on their Chase Visa card. Demanding immediate refund.",
          questions: {
            category: {
              type: "choice",
              instructions: "Classify incoming customer inquiry intent",
              options: [
                { id: "billing_refund", description: "Customer requesting financial reimbursement or reporting duplicate charge" },
                { id: "technical_support", description: "Application error, bug report, or outage issue" },
                { id: "account_access", description: "Password reset, login failure, or MFA problem" },
              ],
            },
          },
        });

        // Warm up
        zev_evaluate_json(sampleRequest);

        const chunks = 10;
        const perChunk = 100;
        const totalIterations = chunks * perChunk;
        let totalElapsedMs = 0;

        for (let c = 0; c < chunks; c++) {
          const chunkStart = performance.now();
          for (let i = 0; i < perChunk; i++) {
            zev_evaluate_json(sampleRequest);
          }
          await new Promise((r) => setTimeout(r, 0));
          totalElapsedMs += performance.now() - chunkStart;
        }

        const avgUs = (totalElapsedMs * 1000) / totalIterations;
        const throughput = Math.round((totalIterations / (totalElapsedMs / 1000)));

        return Response.json(
          {
            benchmark: "Zev WASM Edge Isolate Performance",
            iterations: totalIterations,
            total_duration_ms: Math.round(totalElapsedMs * 100) / 100,
            avg_latency_us: Math.round(avgUs * 10) / 10,
            throughput_decisions_per_sec: throughput,
            sample_verification: JSON.parse(zev_evaluate_json(sampleRequest)),
          },
          {
            headers: {
              ...corsHeaders,
              "x-zev-avg-latency-us": avgUs.toFixed(1),
              "x-zev-throughput": throughput.toString(),
            },
          }
        );
      }

      // 3. Jev / Clef / TypeSafe SystemOne Wire Format
      if ((path === "/v1/system_one" || path === "/system_one") && method === "POST") {
        const bodyText = await request.text();
        const t0 = performance.now();
        const resultJson = zev_evaluate_system_one_json(bodyText);
        const durationUs = Math.round((performance.now() - t0) * 1000);

        return new Response(resultJson, {
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
            "x-zev-execution-us": durationUs.toString(),
          },
        });
      }

      // 4. Standard Zev Evaluate Format
      if ((path === "/evaluate" || path === "/") && method === "POST") {
        const bodyText = await request.text();
        const t0 = performance.now();
        const resultJson = zev_evaluate_json(bodyText);
        const durationUs = Math.round((performance.now() - t0) * 1000);

        return new Response(resultJson, {
          headers: {
            ...corsHeaders,
            "Content-Type": "application/json",
            "x-zev-execution-us": durationUs.toString(),
          },
        });
      }

      return new Response(JSON.stringify({ error: `Not found: ${method} ${path}` }), {
        status: 404,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    } catch (err: any) {
      return new Response(
        JSON.stringify({
          error: "Zev WASM evaluation failure",
          message: err?.message || String(err),
        }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }
  },
};
