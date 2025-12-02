import datetime
import zipfile
from pathlib import Path
from typing import Optional


def zip_command(output_file: Optional[str] = None) -> int:
    """
    Create a zip file containing the context of the current directory.
    
    This command captures all files in the current directory.
    The resulting zip file can be extracted to restore the complete context.
    
    Args:
        output_file: Name of the output zip file (default: "upsonic_context_<timestamp>.zip")
    
    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        # Lazy import printer functions
        from upsonic.cli.printer import (
            print_error,
            print_success,
            print_info,
        )
        
        # Get current directory
        current_dir = Path.cwd()
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"upsonic_context_{timestamp}.zip"
        
        # Ensure .zip extension
        if not output_file.endswith('.zip'):
            output_file += '.zip'
        
        output_path = current_dir / output_file
        
        print_info(f"Creating context zip file: {output_file}")
        print_info(f"Scanning directory: {current_dir}")
        
        # Collect files to include
        files_to_zip = []
        total_size = 0
        
        for item in current_dir.rglob('*'):
            if item.is_file() and item != output_path:
                try:
                    size = item.stat().st_size
                    files_to_zip.append(item)
                    total_size += size
                except (PermissionError, OSError) as e:
                    print_info(f"Skipping {item.name}: {str(e)}")
        
        if not files_to_zip:
            print_error("No files found to include in the zip")
            return 1
        
        print_info(f"Found {len(files_to_zip)} files ({total_size / 1024 / 1024:.2f} MB)")
        
        # Create the zip file
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files
            for file_path in files_to_zip:
                relative_path = file_path.relative_to(current_dir)
                try:
                    zipf.write(file_path, arcname=relative_path)
                except Exception as e:
                    print_info(f"Warning: Could not add {relative_path}: {str(e)}")
        
        # Print success
        final_size = output_path.stat().st_size
        print_success(f"Context zip created successfully: {output_file}")
        print_info(f"Archive size: {final_size / 1024 / 1024:.2f} MB")
        print_info(f"Files included: {len(files_to_zip)}")
        print_info(f"Extract this file to restore the complete context")
        
        return 0
        
    except KeyboardInterrupt:
        from upsonic.cli.printer import print_cancelled
        print_cancelled()
        return 1
    except Exception as e:
        from upsonic.cli.printer import print_error
        print_error(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

