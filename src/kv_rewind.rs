use crate::error::{Result, ZevError};
use std::collections::HashMap;

/// A fixed-size page in the context arena.
pub const DEFAULT_PAGE_SIZE: usize = 16;

/// Page table tracking for a document or state's resident context.
#[derive(Debug, Clone)]
pub struct BlockTable {
    pub key: String,
    pub page_ids: Vec<usize>,
    pub token_count: usize,
    pub checkpoint_tokens: usize,
    pub checkpoint_page_count: usize,
}

/// Paged context arena managing allocation, retention, and KV rewind.
///
/// Implements Quail's PageArena principles: fixed-size pages, zero reallocation,
/// block tables, and rewindable query suffix evaluation.
#[derive(Debug)]
pub struct PagedContextArena {
    page_size: usize,
    total_pages: usize,
    free_pages: Vec<usize>,
    block_tables: HashMap<String, BlockTable>,
    pub alloc_count: usize,
    pub rewind_count: usize,
    pub free_count: usize,
}

impl PagedContextArena {
    pub fn new(total_pages: usize, page_size: usize) -> Self {
        let mut free_pages = Vec::with_capacity(total_pages);
        for i in (0..total_pages).rev() {
            free_pages.push(i);
        }

        Self {
            page_size,
            total_pages,
            free_pages,
            block_tables: HashMap::new(),
            alloc_count: 0,
            rewind_count: 0,
            free_count: 0,
        }
    }

    #[inline]
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    #[inline]
    pub fn free_pages_count(&self) -> usize {
        self.free_pages.len()
    }

    #[inline]
    pub fn total_pages(&self) -> usize {
        self.total_pages
    }

    pub fn pages_needed(&self, tokens: usize) -> usize {
        if tokens == 0 {
            0
        } else {
            tokens.div_ceil(self.page_size)
        }
    }

    /// Allocates resident prefix pages for a key.
    pub fn allocate_prefix(&mut self, key: &str, token_count: usize) -> Result<&BlockTable> {
        if self.block_tables.contains_key(key) {
            return Err(ZevError::Internal(format!(
                "Key '{key}' is already resident in arena"
            )));
        }

        let needed = self.pages_needed(token_count);
        if needed > self.free_pages.len() {
            return Err(ZevError::Internal(format!(
                "Arena OOM: needed {} pages, but only {} free",
                needed,
                self.free_pages.len()
            )));
        }

        let mut page_ids = Vec::with_capacity(needed);
        for _ in 0..needed {
            page_ids.push(self.free_pages.pop().unwrap());
        }

        self.alloc_count += needed;
        let table = BlockTable {
            key: key.to_string(),
            page_ids,
            token_count,
            checkpoint_tokens: token_count,
            checkpoint_page_count: needed,
        };

        self.block_tables.insert(key.to_string(), table);
        Ok(self.block_tables.get(key).unwrap())
    }

    /// Checkpoints the current context length for a key (saving prefix watermark).
    pub fn checkpoint(&mut self, key: &str) -> Result<()> {
        let table = self
            .block_tables
            .get_mut(key)
            .ok_or_else(|| ZevError::Internal(format!("Key '{key}' not found in arena")))?;
        table.checkpoint_tokens = table.token_count;
        table.checkpoint_page_count = table.page_ids.len();
        Ok(())
    }

    /// Appends query tokens to a resident state.
    pub fn append_suffix(&mut self, key: &str, suffix_tokens: usize) -> Result<usize> {
        let needed_total = {
            let table = self
                .block_tables
                .get(key)
                .ok_or_else(|| ZevError::Internal(format!("Key '{key}' not found in arena")))?;
            table.token_count + suffix_tokens
        };

        let new_pages_needed = self.pages_needed(needed_total);
        let table = self.block_tables.get_mut(key).unwrap();
        let current_pages = table.page_ids.len();

        if new_pages_needed > current_pages {
            let delta = new_pages_needed - current_pages;
            if delta > self.free_pages.len() {
                return Err(ZevError::Internal(format!(
                    "Arena OOM when appending suffix: needed {} pages, but only {} free",
                    delta,
                    self.free_pages.len()
                )));
            }

            for _ in 0..delta {
                table.page_ids.push(self.free_pages.pop().unwrap());
            }
            self.alloc_count += delta;
        }

        table.token_count = needed_total;
        Ok(table.token_count)
    }

    /// Rewinds a state back to its last checkpoint, releasing suffix pages back to free list.
    pub fn rewind(&mut self, key: &str) -> Result<usize> {
        let table = self
            .block_tables
            .get_mut(key)
            .ok_or_else(|| ZevError::Internal(format!("Key '{key}' not found in arena")))?;

        let current_pages = table.page_ids.len();
        let target_pages = table.checkpoint_page_count;

        if current_pages > target_pages {
            let to_free = current_pages - target_pages;
            for _ in 0..to_free {
                let page_id = table.page_ids.pop().unwrap();
                self.free_pages.push(page_id);
            }
            self.free_count += to_free;
        }

        table.token_count = table.checkpoint_tokens;
        self.rewind_count += 1;
        Ok(table.token_count)
    }

    /// Frees an entire resident key from the arena.
    pub fn free_key(&mut self, key: &str) -> Result<usize> {
        let table = self
            .block_tables
            .remove(key)
            .ok_or_else(|| ZevError::Internal(format!("Key '{key}' not found in arena")))?;

        let count = table.page_ids.len();
        for page_id in table.page_ids {
            self.free_pages.push(page_id);
        }
        self.free_count += count;
        Ok(count)
    }

    /// Helper that appends a suffix, runs an evaluation closure, and automatically rewinds.
    pub fn with_rewind<F, R>(&mut self, key: &str, suffix_tokens: usize, f: F) -> Result<R>
    where
        F: FnOnce(&BlockTable) -> Result<R>,
    {
        self.append_suffix(key, suffix_tokens)?;
        let res = {
            let table = self.block_tables.get(key).unwrap();
            f(table)
        };
        self.rewind(key)?;
        res
    }
}
