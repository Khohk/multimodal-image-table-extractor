"""
Table Utilities - Convert JSON tables to structured text formats
Supports: Markdown, CSV, Plain Text
"""

import json
import pandas as pd
from io import StringIO
from pathlib import Path
from typing import Optional, Dict, List


def load_table_json(json_path: str) -> Optional[Dict]:
    """
    Load table JSON file safely
    
    Args:
        json_path: Path to table JSON file
        
    Returns:
        Dict with table data or None if error
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            table_data = json.load(f)
        return table_data
    except FileNotFoundError:
        print(f"❌ File not found: {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {json_path}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading {json_path}: {e}")
        return None


def table_to_dataframe(table_data: Dict) -> Optional[pd.DataFrame]:
    """
    Convert table JSON to pandas DataFrame
    
    Supports two formats:
    1. {'headers': [...], 'rows': [[...], ...]}
    2. {'data': [{'col1': val1, ...}, ...]}
    
    Args:
        table_data: Table data from JSON
        
    Returns:
        DataFrame or None if conversion fails
    """
    try:
        # Format 1: headers + rows
        if "headers" in table_data and "rows" in table_data:
            headers = table_data["headers"]
            rows = table_data["rows"]
            
            if not rows:
                return None
            
            df = pd.DataFrame(rows, columns=headers)
            return df
        
        # Format 2: data (list of dicts)
        elif "data" in table_data:
            data = table_data["data"]
            if not data:
                return None
            df = pd.DataFrame(data)
            return df
        
        else:
            print("❌ Unknown table format (no 'headers'/'rows' or 'data' key)")
            return None
            
    except Exception as e:
        print(f"❌ Error converting to DataFrame: {e}")
        return None


def table_to_text(json_path: str, format: str = "markdown", 
                  max_rows: Optional[int] = None) -> str:
    """
    Convert table JSON to structured text
    
    Args:
        json_path: Path to table JSON file
        format: Output format ('markdown', 'csv', 'plain')
        max_rows: Maximum rows to include (None = all rows)
        
    Returns:
        Formatted table as string
    """
    # Load table
    table_data = load_table_json(json_path)
    if not table_data:
        return f"⚠️ Table file unreadable: {json_path}"
    
    # Extract caption
    caption = table_data.get("caption", "No caption")
    
    # Convert to DataFrame
    df = table_to_dataframe(table_data)
    if df is None or df.empty:
        return f"{caption}\n(No data rows found.)"
    
    # Limit rows if specified
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
        truncated_note = f"\n(Showing {max_rows} of {len(df)} rows)"
    else:
        truncated_note = ""
    
    # Format table
    try:
        if format == "csv":
            buf = StringIO()
            df.to_csv(buf, index=False)
            table_text = buf.getvalue()
        
        elif format == "plain":
            table_text = df.to_string(index=False)
        
        else:  # markdown (default)
            table_text = df.to_markdown(index=False)
        
        return f"{caption}\n\n{table_text}{truncated_note}"
    
    except Exception as e:
        print(f"❌ Error formatting table: {e}")
        return f"{caption}\n(Error formatting table: {e})"


def get_table_summary_stats(json_path: str) -> Dict:
    """
    Get basic statistics about table structure
    
    Args:
        json_path: Path to table JSON
        
    Returns:
        Dict with stats (rows, columns, caption length, etc.)
    """
    table_data = load_table_json(json_path)
    if not table_data:
        return {"error": "Cannot load table"}
    
    df = table_to_dataframe(table_data)
    
    stats = {
        "caption": table_data.get("caption", ""),
        "caption_length": len(table_data.get("caption", "")),
        "rows": len(df) if df is not None else 0,
        "columns": len(df.columns) if df is not None else 0,
        "has_hierarchical_headers": table_data.get("metadata", {}).get("has_hierarchical_headers", False),
        "extraction_method": table_data.get("extraction_method", "unknown")
    }
    
    return stats


# Example usage
if __name__ == "__main__":
    # Test with a sample table
    test_json = "extracted_content/tables/page3_table1.json"
    
    if Path(test_json).exists():
        print("📊 Table Stats:")
        print(json.dumps(get_table_summary_stats(test_json), indent=2))
        
        print("\n📝 Markdown Format:")
        print(table_to_text(test_json, format="markdown", max_rows=5))
        
        print("\n📄 CSV Format:")
        print(table_to_text(test_json, format="csv", max_rows=5))
    else:
        print(f"❌ Test file not found: {test_json}")