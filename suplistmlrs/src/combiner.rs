// Copyright (c) 2025 Chad Voegele
//
// Licensed under the GPL-2.0 License. See LICENSE file for full license information.

use crate::model::MetaRow;
use std::collections::HashMap;
use std::error::Error;

fn get_grouping_key(name: &str) -> String {
    if name.is_empty() {
        return "".to_string();
    }
    let lower = name.to_lowercase();

    if lower.ends_with("berries") && lower.len() > 7 {
        return lower[..lower.len() - 7].to_string() + "berry";
    }
    if lower.ends_with("ches") && lower.len() > 2 {
        return lower[..lower.len() - 4].to_string() + "ch";
    }
    if lower.ends_with("shes") && lower.len() > 2 {
        return lower[..lower.len() - 4].to_string() + "sh";
    }
    if lower.ends_with("sses") && lower.len() > 2 {
        return lower[..lower.len() - 4].to_string() + "ss";
    }
    if lower.ends_with("xes") && lower.len() > 2 {
        return lower[..lower.len() - 3].to_string() + "x";
    }
    if lower.ends_with("ies") && lower.len() > 3 {
        return lower[..lower.len() - 3].to_string() + "y";
    }
    if lower.ends_with('s') && lower.len() > 1 {
        if let Some(prev_char) = lower.chars().nth(lower.len() - 2) {
            if prev_char != 's' {
                return lower[..lower.len() - 1].to_string();
            }
        } else {
            return lower[..lower.len() - 1].to_string();
        }
    }
    lower
}

pub fn combine(rows: Vec<MetaRow>) -> Result<Vec<MetaRow>, Box<dyn Error>> {
    if rows.is_empty() {
        return Ok(vec![]);
    }

    let mut named_rows: Vec<MetaRow> = Vec::new();
    let mut no_name_rows: Vec<MetaRow> = Vec::new();
    for row in rows {
        if row.name.is_empty() {
            no_name_rows.push(row);
        } else {
            named_rows.push(row);
        }
    }

    let mut grouped_data: HashMap<String, (MetaRow, Vec<String>)> = HashMap::new();

    for row in named_rows {
        let key = get_grouping_key(&row.name);

        let entry = grouped_data
            .entry(key)
            .or_insert_with(|| (row.clone(), Vec::new()));
        entry.1.push(row.text);
    }

    let mut combined_rows: Vec<MetaRow> = grouped_data
        .into_values()
        .map(|(mut first_row, texts)| {
            first_row.text = texts.join(" + ");
            first_row
        })
        .collect();

    combined_rows.append(&mut no_name_rows);

    combined_rows.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(combined_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MetaRow;

    fn _meta_row(text: &str, name: &str, category: &str) -> MetaRow {
        MetaRow {
            name: name.to_string(),
            category: category.to_string(),
            text: text.to_string(),
        }
    }

    #[test]
    fn test_combine() {
        let input = vec![
            _meta_row("1 pear", "pear", "fruit"),
            _meta_row("salt", "salt", "condiments"),
            _meta_row("apple", "apple", "fruit"),
            _meta_row("1 tsp salt", "salt", "condiments"),
        ];

        let result = combine(input).unwrap();

        let expected_rows = vec![
            _meta_row("apple", "apple", "fruit"),
            _meta_row("1 pear", "pear", "fruit"),
            _meta_row("salt + 1 tsp salt", "salt", "condiments"),
        ];

        assert_eq!(result, expected_rows);
    }

    #[test]
    fn test_combine_no_name() {
        let input = vec![
            _meta_row("salt", "salt", "condiments"),
            _meta_row("saltines", "", "condiments"),
        ];

        let result = combine(input).unwrap();

        let expected_rows = vec![
            _meta_row("saltines", "", "condiments"),
            _meta_row("salt", "salt", "condiments"),
        ];

        assert_eq!(result, expected_rows);
    }

    #[test]
    fn test_combine_all_no_name() {
        let input = vec![
            _meta_row("bar", "", "produce"),
            _meta_row("2 hersheys kiss", "", "produce"),
        ];

        let result = combine(input).unwrap();

        let expected_rows = vec![
            _meta_row("bar", "", "produce"),
            _meta_row("2 hersheys kiss", "", "produce"),
        ];

        assert_eq!(result, expected_rows);
    }

    #[test]
    fn test_combine_purals() {
        let input = vec![
            _meta_row("peach", "peach", "produce"),
            _meta_row("2 peaches", "peaches", "produce"),
        ];

        let result = combine(input).unwrap();

        let expected_rows = vec![_meta_row("peach + 2 peaches", "peach", "produce")];

        assert_eq!(result, expected_rows);
    }

    #[test]
    fn test_combine_purals_stem() {
        let input = vec![
            _meta_row("1 blueberry", "blueberry", "produce"),
            _meta_row("2 blueberries", "blueberries", "produce"),
        ];

        let result = combine(input).unwrap();

        let expected_rows = vec![_meta_row(
            "1 blueberry + 2 blueberries",
            "blueberry",
            "produce",
        )];

        assert_eq!(result, expected_rows);
    }

    #[test]
    fn test_combine_category_from_first() {
        let input = vec![
            _meta_row("1 apple", "apple", "Red Apples"),
            _meta_row("2 apples", "apples", "Green Apples"),
        ];
        let result = combine(input).unwrap();
        let expected_rows = vec![_meta_row("1 apple + 2 apples", "apple", "Red Apples")];
        assert_eq!(result, expected_rows);
    }

    #[test]
    fn test_combine_empty() {
        let input: Vec<MetaRow> = vec![];
        let result = combine(input).unwrap();
        let expected_rows: Vec<MetaRow> = vec![];
        assert_eq!(result, expected_rows);
    }
}
