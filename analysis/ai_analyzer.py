"""
AI Analyzer - Analyze figures and tables using Gemini Vision API
Fixed: Handle ragStoreName parameter issue
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Literal
import google.generativeai as genai
from PIL import Image
import io

from .config import AnalysisConfig
from utils.table_utils import table_to_text, get_table_summary_stats
from utils.print_helper import safe_print as print
from utils.print_helper import print_ok, print_error, print_warning, print_info


# Configure Gemini
genai.configure(api_key=AnalysisConfig.GEMINI_API_KEY)


class AIAnalyzer:
    """Main analyzer class for figures and tables"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize analyzer
        
        Args:
            model_name: Gemini model name (default from config)
        """
        self.model_name = model_name or AnalysisConfig.GEMINI_MODEL
        self.model = genai.GenerativeModel(self.model_name)
        self.request_times = []  # Track request times for rate limiting
        
        print(f"🤖 Initialized AIAnalyzer with model: {self.model_name}")
    
    async def _wait_for_rate_limit(self):
        """Implement rate limiting (15 RPM for Gemini)"""
        now = time.time()
        
        # Remove requests older than 60 seconds
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # If we've hit the limit, wait
        if len(self.request_times) >= AnalysisConfig.RATE_LIMIT_RPM:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                print(f"⏳ Rate limit reached. Waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                self.request_times = []
        
        self.request_times.append(now)
    
    def _load_image_as_pil(self, image_path: str) -> Image.Image:
        """
        Load image as PIL Image object
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        try:
            img = Image.open(image_path)
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            return img
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {e}")
    
    async def analyze_figure(self, image_path: str, caption: str = "",
                            max_retries: Optional[int] = None) -> Dict:
        """
        Analyze a figure/image using Gemini Vision
        
        Args:
            image_path: Path to image file
            caption: Figure caption
            max_retries: Number of retry attempts
            
        Returns:
            Dict with analysis results
        """
        max_retries = max_retries or AnalysisConfig.MAX_RETRIES
        
        for attempt in range(max_retries):
            try:
                await self._wait_for_rate_limit()
                
                # Method 1: Try with PIL Image directly (recommended)
                try:
                    img = self._load_image_as_pil(image_path)
                    
                    # Generate prompt
                    prompt = AnalysisConfig.FIGURE_PROMPT_TEMPLATE.format(
                        caption=caption or "No caption provided"
                    )
                    
                    # Analyze with PIL Image
                    response = self.model.generate_content([prompt, img])
                    
                    return {
                        "path": image_path,
                        "caption": caption,
                        "type": "figure",
                        "summary": response.text,
                        "model": self.model_name,
                        "status": "success"
                    }
                
                except Exception as pil_error:
                    # Method 2: Fallback to file upload with display_name
                    print(f"⚠️ PIL method failed, trying file upload: {pil_error}")
                    
                    img_file = genai.upload_file(
                        image_path,
                        display_name=Path(image_path).name
                    )
                    
                    # Generate prompt
                    prompt = AnalysisConfig.FIGURE_PROMPT_TEMPLATE.format(
                        caption=caption or "No caption provided"
                    )
                    
                    # Analyze
                    response = self.model.generate_content([prompt, img_file])
                    
                    # Clean up uploaded file
                    try:
                        genai.delete_file(img_file.name)
                    except:
                        pass
                    
                    return {
                        "path": image_path,
                        "caption": caption,
                        "type": "figure",
                        "summary": response.text,
                        "model": self.model_name,
                        "status": "success"
                    }
                
            except Exception as e:
                error_msg = str(e)
                print(f"⚠️ [Retry {attempt + 1}/{max_retries}] Error analyzing {image_path}: {error_msg}")
                
                if attempt < max_retries - 1:
                    wait_time = AnalysisConfig.RETRY_DELAY_BASE ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    return {
                        "path": image_path,
                        "caption": caption,
                        "type": "figure",
                        "summary": None,
                        "error": error_msg,
                        "status": "failed"
                    }
    
    async def analyze_table(self, json_path: str, caption: str = "",
                           summary_mode: Literal["text", "json", "detailed"] = "text",
                           max_retries: Optional[int] = None) -> Dict:
        """
        Analyze a table using Gemini
        
        Args:
            json_path: Path to table JSON file
            caption: Table caption
            summary_mode: Output format ('text', 'json', 'detailed')
            max_retries: Number of retry attempts
            
        Returns:
            Dict with analysis results
        """
        max_retries = max_retries or AnalysisConfig.MAX_RETRIES
        
        # Get table stats
        stats = get_table_summary_stats(json_path)
        if "error" in stats:
            return {
                "path": json_path,
                "caption": caption,
                "type": "table",
                "summary": None,
                "error": stats["error"],
                "status": "failed"
            }
        
        # Convert table to text
        table_text = table_to_text(
            json_path,
            format=AnalysisConfig.TABLE_FORMAT,
            max_rows=AnalysisConfig.TABLE_MAX_ROWS
        )
        
        for attempt in range(max_retries):
            try:
                await self._wait_for_rate_limit()
                
                # Select prompt based on mode
                if summary_mode == "json":
                    prompt = AnalysisConfig.TABLE_JSON_PROMPT_TEMPLATE.format(
                        caption=caption or "No caption",
                        table_text=table_text
                    )
                else:
                    prompt = AnalysisConfig.TABLE_TEXT_PROMPT_TEMPLATE.format(
                        caption=caption or "No caption",
                        table_text=table_text
                    )
                
                # Analyze
                response = self.model.generate_content(prompt)
                summary = response.text
                
                # Try to parse JSON if requested
                parsed_json = None
                if summary_mode == "json":
                    try:
                        # Remove markdown code blocks if present
                        clean_text = summary.strip()
                        if clean_text.startswith("```"):
                            clean_text = clean_text.split("```")[1]
                            if clean_text.startswith("json"):
                                clean_text = clean_text[4:]
                        parsed_json = json.loads(clean_text)
                    except json.JSONDecodeError:
                        print(f"⚠️ Could not parse JSON response for {json_path}")
                
                return {
                    "path": json_path,
                    "caption": caption,
                    "type": "table",
                    "summary": summary,
                    "summary_json": parsed_json,
                    "summary_mode": summary_mode,
                    "model": self.model_name,
                    "stats": stats,
                    "status": "success"
                }
                
            except Exception as e:
                print(f"⚠️ [Retry {attempt + 1}/{max_retries}] Error analyzing {json_path}: {e}")
                
                if attempt < max_retries - 1:
                    wait_time = AnalysisConfig.RETRY_DELAY_BASE ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    return {
                        "path": json_path,
                        "caption": caption,
                        "type": "table",
                        "summary": None,
                        "error": str(e),
                        "status": "failed"
                    }
    
    async def analyze_batch(self, items: List[Dict], 
                           batch_size: Optional[int] = None) -> List[Dict]:
        """
        Analyze multiple figures/tables in batches
        
        Args:
            items: List of dicts with keys:
                   - 'path': file path
                   - 'type': 'figure' or 'table'
                   - 'caption': caption text
                   - 'summary_mode': (for tables only)
            batch_size: Batch size for processing
            
        Returns:
            List of analysis results
        """
        batch_size = batch_size or AnalysisConfig.BATCH_SIZE
        results = []
        
        print(f"🚀 Starting batch analysis of {len(items)} items...")
        print(f"📦 Batch size: {batch_size}")
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            print(f"\n📊 Processing batch {i // batch_size + 1} ({len(batch)} items)...")
            
            tasks = []
            for item in batch:
                if item['type'] == 'figure':
                    task = self.analyze_figure(
                        item['path'],
                        item.get('caption', '')
                    )
                else:  # table
                    task = self.analyze_table(
                        item['path'],
                        item.get('caption', ''),
                        item.get('summary_mode', 'text')
                    )
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions in results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    print_error(f"Exception in batch item {j}: {result}")
                    batch_results[j] = {
                        "path": batch[j].get('path', 'unknown'),
                        "type": batch[j].get('type', 'unknown'),
                        "error": str(result),
                        "status": "failed"
                    }
            
            results.extend(batch_results)
            
            print(f"✅ Batch {i // batch_size + 1} complete")
        
        # Print summary
        success_count = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'success')
        failed_count = len(results) - success_count
        
        print(f"\n🎉 Batch analysis complete!")
        print(f"✅ Success: {success_count}/{len(results)}")
        if failed_count > 0:
            print(f"❌ Failed: {failed_count}/{len(results)}")
        
        return results


# Convenience function for single file analysis
async def analyze_single(path: str, element_type: str = "auto",
                        caption: str = "", summary_mode: str = "text") -> Dict:
    """
    Quick analysis of a single figure or table
    
    Args:
        path: File path
        element_type: 'figure', 'table', or 'auto' (detect from extension)
        caption: Caption text
        summary_mode: Summary mode for tables
        
    Returns:
        Analysis result dict
    """
    analyzer = AIAnalyzer()
    
    # Auto-detect type
    if element_type == "auto":
        ext = Path(path).suffix.lower()
        if ext in ['.png', '.jpg', '.jpeg']:
            element_type = 'figure'
        elif ext == '.json':
            element_type = 'table'
        else:
            raise ValueError(f"Cannot auto-detect type for extension: {ext}")
    
    if element_type == 'figure':
        return await analyzer.analyze_figure(path, caption)
    else:
        return await analyzer.analyze_table(path, caption, summary_mode)


# Example usage
if __name__ == "__main__":
    async def test():
        # Test figure analysis
        result = await analyze_single(
            "extracted_content/figures/page1_fig1.png",
            element_type="figure",
            caption="Figure 1: Sample architecture diagram"
        )
        print("\n📊 Figure Analysis Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Test table analysis
        result = await analyze_single(
            "extracted_content/tables/page3_table1.json",
            element_type="table",
            caption="Table 1: Performance comparison",
            summary_mode="json"
        )
        print("\n📊 Table Analysis Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    asyncio.run(test())