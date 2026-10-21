//! JSON entries, on `jsonc-parser`'s spans: the properties of the root
//! object, or the elements of the root array; comments and trailing commas
//! are accepted, as in `tsconfig.json`.

use jsonc_parser::ast::{ObjectPropName, Value};
use jsonc_parser::common::Ranged;
use jsonc_parser::{CollectOptions, ParseOptions, parse_to_ast};

use super::Entry;
use crate::chunk::Error;

pub(super) fn entries(src: &str) -> Result<Vec<Entry>, Error> {
    let parsed = parse_to_ast(src, &CollectOptions::default(), &ParseOptions::default())?;
    Ok(match parsed.value {
        None => Vec::new(),
        Some(Value::Object(object)) => object.properties.iter().map(property).collect(),
        Some(Value::Array(array)) => elements(&array.elements),
        Some(scalar) => {
            let range = scalar.range();
            vec![Entry::key(
                "value".to_owned(),
                range.start..range.end,
                Vec::new(),
            )]
        }
    })
}

fn property(prop: &jsonc_parser::ast::ObjectProp) -> Entry {
    let name = match &prop.name {
        ObjectPropName::String(literal) => literal.value.as_ref(),
        ObjectPropName::Word(word) => word.value,
    };
    let range = prop.range();
    Entry::key(
        name.to_owned(),
        range.start..range.end,
        children(&prop.value),
    )
}

fn elements(values: &[Value]) -> Vec<Entry> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let range = value.range();
            Entry::element(index, range.start..range.end, children(value))
        })
        .collect()
}

fn children(value: &Value) -> Vec<Entry> {
    match value {
        Value::Object(object) => object.properties.iter().map(property).collect(),
        Value::Array(array) => elements(&array.elements),
        _ => Vec::new(),
    }
}
