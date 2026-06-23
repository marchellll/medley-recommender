use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct CursorPage<T> {
    pub items: Vec<T>,
    pub limit: u32,
    pub next_last_id: Option<String>,
    pub next_last_rank: Option<f64>,
    pub has_more: bool,
    pub total: i64,
}

impl<T> CursorPage<T> {
    pub fn from_rows(
        mut rows: Vec<T>,
        limit: u32,
        total: i64,
        last_id_fn: impl Fn(&T) -> &str,
        last_rank_fn: Option<impl Fn(&T) -> f64>,
    ) -> Self {
        let has_more = rows.len() > limit as usize;
        if has_more {
            rows.truncate(limit as usize);
        }
        let (next_last_id, next_last_rank) = if has_more {
            if let Some(last) = rows.last() {
                let rank = last_rank_fn.map(|f| f(last));
                (Some(last_id_fn(last).to_string()), rank)
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        Self {
            items: rows,
            limit,
            next_last_id,
            next_last_rank,
            has_more,
            total,
        }
    }
}

pub fn clamp_limit(limit: Option<u32>, default: u32, max: u32) -> u32 {
    limit.unwrap_or(default).min(max).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_limit_uses_default_and_bounds() {
        assert_eq!(clamp_limit(None, 10, 50), 10);
        assert_eq!(clamp_limit(Some(100), 10, 50), 50);
        assert_eq!(clamp_limit(Some(0), 10, 50), 1);
    }
}
