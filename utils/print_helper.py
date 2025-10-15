"""
Cross-platform print helper - handles Unicode emoji on Windows
"""

import sys
import platform
import builtins # BỔ SUNG: Import module builtins để truy cập hàm print gốc


def safe_print(*args, **kwargs):
    """
    Print with safe Unicode handling for Windows
    Replaces emoji with ASCII equivalents on Windows terminals
    """
    # Check if running on Windows
    is_windows = platform.system() == 'Windows'
    
    if is_windows:
        # Try to set UTF-8 encoding for Windows terminal
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            # Python < 3.7 doesn't have reconfigure
            pass
    
    try:
        # SỬA LỖI: Gọi hàm print gốc từ builtins để tránh đệ quy
        builtins.print(*args, **kwargs) 
    except UnicodeEncodeError:
        # Fallback: replace problematic characters
        message = ' '.join(str(arg) for arg in args)
        safe_message = replace_emoji(message)
        builtins.print(safe_message, **kwargs) # SỬA LỖI: Gọi builtins.print


def replace_emoji(text: str) -> str:
    # ... (Giữ nguyên hàm replace_emoji)
    emoji_map = {
        '✅': '[OK]',
        '❌': '[ERROR]',
        # ... (các mục khác)
        '👉': '>>',
        '👋': '[BYE]',
        '⏸️': '[PAUSE]',
    }
    
    for emoji, replacement in emoji_map.items():
        text = text.replace(emoji, replacement)
    
    return text


# Convenience functions
def print_ok(message):
    """Print success message"""
    safe_print(f"[OK] {message}")


def print_error(message):
    """Print error message"""
    safe_print(f"[ERROR] {message}")


def print_warning(message):
    """Print warning message"""
    safe_print(f"[WARNING] {message}")


def print_info(message):
    """Print info message"""
    safe_print(f"[INFO] {message}")


def print_separator(title: str, width: int = 70):
    """Print a nice separator"""
    safe_print("\n" + "=" * width)
    safe_print(f"  {title}")
    safe_print("=" * width)


# XÓA DÒNG GÂY LỖI: print = safe_print
# KHÔNG CÓ DÒNG NÀO Ở ĐÂY NỮA