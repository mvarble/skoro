//! The data behind the examples' figures: one row's truth, start, mode and marginal covariances,
//! and every normalized error, written as JSON when `SKORO_FIGURES` names a directory.
//! `tools/figures` draws them.

use std::fmt::Write as _;

use super::report::Result;

/// A JSON value, written by hand: the examples need nothing more than numbers, arrays and
/// objects.
pub enum Json {
    Number(f64),
    Array(Vec<Json>),
    Object(Vec<(&'static str, Json)>),
}

impl Json {
    pub fn numbers(values: &[f64]) -> Json {
        Json::Array(values.iter().copied().map(Json::Number).collect())
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Number(x) if x.is_finite() => {
                let _ = write!(out, "{x}");
            }
            Json::Number(_) => out.push_str("null"),
            Json::Array(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Json::Object(fields) => {
                out.push('{');
                for (i, (key, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    let _ = write!(out, "\"{key}\":");
                    value.write(out);
                }
                out.push('}');
            }
        }
    }
}

/// Writes `data` as `<name>.json` in the directory `SKORO_FIGURES` names, if it names one.
pub fn write(name: &str, data: &Json) -> Result {
    let Some(dir) = std::env::var_os("SKORO_FIGURES") else {
        return Ok(());
    };
    let mut text = String::new();
    data.write(&mut text);
    let path = std::path::Path::new(&dir).join(format!("{name}.json"));
    std::fs::write(&path, text)?;
    println!("wrote {}", path.display());
    Ok(())
}

/// Whether the figures' data is asked for, so that an example gathers it only then.
pub fn wanted() -> bool {
    std::env::var_os("SKORO_FIGURES").is_some()
}

/// The position `(x, y, z)` of row `r` of a value stored `storage` to a row, its last three
/// entries: a pose's position or a landmark itself.
pub fn position(data: &[f64], storage: usize, r: usize) -> Json {
    let row = &data[r * storage..(r + 1) * storage];
    Json::numbers(&row[storage - 3..])
}

#[cfg(test)]
mod tests {
    use super::Json;

    #[test]
    fn json_is_written_compactly() {
        let data = Json::Object(vec![
            ("a", Json::numbers(&[1.0, 0.5])),
            ("b", Json::Number(f64::NAN)),
        ]);
        let mut out = String::new();
        data.write(&mut out);
        assert_eq!(out, r#"{"a":[1,0.5],"b":null}"#);
    }
}
