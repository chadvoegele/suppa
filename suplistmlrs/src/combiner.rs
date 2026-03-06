// Copyright (c) 2025 Chad Voegele
//
// Licensed under the GPL-2.0 License. See LICENSE file for full license information.

use crate::model::MetaRow;
use std::collections::HashMap;
use std::error::Error;

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct GroupedMetaRow {
    pub text: Vec<String>,
    pub category: String,
    pub name: String,
    pub qty: Vec<String>,
    pub unit: Vec<String>,
}

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

pub fn combine(rows: Vec<MetaRow>) -> Result<Vec<GroupedMetaRow>, Box<dyn Error>> {
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

    let mut grouped_data: HashMap<String, GroupedMetaRow> = HashMap::new();

    for row in named_rows {
        let key = get_grouping_key(&row.name);

        let entry = grouped_data.entry(key).or_insert_with(|| GroupedMetaRow {
            name: row.name.clone(),
            category: row.category.clone(),
            text: Vec::new(),
            qty: Vec::new(),
            unit: Vec::new(),
        });
        entry.text.push(row.text);
        entry.qty.push(row.qty);
        entry.unit.push(row.unit);
    }

    let mut combined_rows: Vec<GroupedMetaRow> = grouped_data.into_values().collect();

    let no_name_grouped: Vec<GroupedMetaRow> = no_name_rows
        .into_iter()
        .map(|row| GroupedMetaRow {
            text: vec![row.text],
            category: row.category,
            name: row.name,
            qty: vec![row.qty],
            unit: vec![row.unit],
        })
        .collect();
    combined_rows.extend(no_name_grouped);

    combined_rows.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(combined_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MetaRow;

    fn _meta_row(text: &str, name: &str, category: &str, qty: &str, unit: &str) -> MetaRow {
        MetaRow {
            name: name.to_string(),
            category: category.to_string(),
            text: text.to_string(),
            qty: qty.to_string(),
            unit: unit.to_string(),
        }
    }

    fn _grouped(
        texts: &[&str],
        name: &str,
        category: &str,
        qtys: &[&str],
        units: &[&str],
    ) -> GroupedMetaRow {
        GroupedMetaRow {
            name: name.to_string(),
            category: category.to_string(),
            text: texts.iter().map(|s| s.to_string()).collect(),
            qty: qtys.iter().map(|s| s.to_string()).collect(),
            unit: units.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn test_combine() {
        let input = vec![
            _meta_row("1 pear", "pear", "fruit", "1", ""),
            _meta_row("salt", "salt", "condiments", "", ""),
            _meta_row("apple", "apple", "fruit", "", ""),
            _meta_row("1 tsp salt", "salt", "condiments", "1", "tsp"),
        ];

        let result = combine(input).unwrap();

        let expected_rows = vec![
            _grouped(&["apple"], "apple", "fruit", &[""], &[""]),
            _grouped(&["1 pear"], "pear", "fruit", &["1"], &[""]),
            _grouped(
                &["salt", "1 tsp salt"],
                "salt",
                "condiments",
                &["", "1"],
                &["", "tsp"],
            ),
        ];

        assert_eq!(result, expected_rows);
    }

    #[test]
    fn test_combine_no_name() {
        let input = vec![
            _meta_row("salt", "salt", "condiments", "", ""),
            _meta_row("saltines", "", "condiments", "", ""),
        ];

        let result = combine(input).unwrap();

        let expected_rows = vec![
            _grouped(&["saltines"], "", "condiments", &[""], &[""]),
            _grouped(&["salt"], "salt", "condiments", &[""], &[""]),
        ];

        assert_eq!(result, expected_rows);
    }

    #[test]
    fn test_combine_all_no_name() {
        let input = vec![
            _meta_row("bar", "", "produce", "", ""),
            _meta_row("2 hersheys kiss", "", "produce", "2", ""),
        ];

        let result = combine(input).unwrap();

        let expected_rows = vec![
            _grouped(&["bar"], "", "produce", &[""], &[""]),
            _grouped(&["2 hersheys kiss"], "", "produce", &["2"], &[""]),
        ];

        assert_eq!(result, expected_rows);
    }

    #[test]
    fn test_combine_purals() {
        let input = vec![
            _meta_row("peach", "peach", "produce", "", ""),
            _meta_row("2 peaches", "peaches", "produce", "2", ""),
        ];

        let result = combine(input).unwrap();

        let expected_rows = vec![_grouped(
            &["peach", "2 peaches"],
            "peach",
            "produce",
            &["", "2"],
            &["", ""],
        )];

        assert_eq!(result, expected_rows);
    }

    #[test]
    fn test_combine_purals_stem() {
        let input = vec![
            _meta_row("1 blueberry", "blueberry", "produce", "1", ""),
            _meta_row("2 blueberries", "blueberries", "produce", "2", ""),
        ];

        let result = combine(input).unwrap();

        let expected_rows = vec![_grouped(
            &["1 blueberry", "2 blueberries"],
            "blueberry",
            "produce",
            &["1", "2"],
            &["", ""],
        )];

        assert_eq!(result, expected_rows);
    }

    #[test]
    fn test_combine_category_from_first() {
        let input = vec![
            _meta_row("1 apple", "apple", "Red Apples", "1", ""),
            _meta_row("2 apples", "apples", "Green Apples", "2", ""),
        ];
        let result = combine(input).unwrap();
        let expected_rows = vec![_grouped(
            &["1 apple", "2 apples"],
            "apple",
            "Red Apples",
            &["1", "2"],
            &["", ""],
        )];
        assert_eq!(result, expected_rows);
    }

    #[test]
    fn test_combine_empty() {
        let input: Vec<MetaRow> = vec![];
        let result = combine(input).unwrap();
        let expected_rows: Vec<GroupedMetaRow> = vec![];
        assert_eq!(result, expected_rows);
    }

    #[test]
    fn test_combine_qty_unit() {
        let input = vec![
            _meta_row(
                "1 cup frozen strawberries, cubed",
                "strawberries",
                "produce",
                "1",
                "cup",
            ),
            _meta_row(
                "1/2 cup frozen strawberries, diced",
                "strawberries",
                "produce",
                "1/2",
                "cup",
            ),
        ];

        let result = combine(input).unwrap();

        let expected_rows = vec![_grouped(
            &[
                "1 cup frozen strawberries, cubed",
                "1/2 cup frozen strawberries, diced",
            ],
            "strawberries",
            "produce",
            &["1", "1/2"],
            &["cup", "cup"],
        )];

        assert_eq!(result, expected_rows);
    }
}
