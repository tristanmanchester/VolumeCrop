import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import logging
import tifffile
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from multiprocessing import cpu_count
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum, auto

class ProcessingError(Exception):
    """Base class for processing-specific exceptions"""
    pass

class DirectoryNotFoundError(ProcessingError):
    """Raised when required directories are not found"""
    pass

class DataLoadError(ProcessingError):
    """Raised when there are issues loading the data"""
    pass

class ProcessingStatus(Enum):
    SUCCESS = auto()
    DIRECTORY_NOT_FOUND = auto()
    DATA_LOAD_FAILED = auto()
    PROCESSING_FAILED = auto()
    SAVE_FAILED = auto()

@dataclass
class ProcessingResult:
    experiment_number: str
    status: ProcessingStatus
    error_message: Optional[str] = None
    processing_time: Optional[float] = None

def load_tiffs_to_ram(directory: Path) -> Dict[int, np.ndarray]:
    """First load all TIFF files into RAM with a progress bar"""
    try:
        # Get and sort files
        tiff_files = sorted(
            [f for f in directory.glob('*.tiff') if f.is_file()],
            key=lambda x: int(x.stem.split('_')[-1])
        )
        if not tiff_files:
            tiff_files = sorted(
                [f for f in directory.glob('*.tif') if f.is_file()],
                key=lambda x: int(x.stem.split('_')[-1])
            )
        
        if not tiff_files:
            raise DataLoadError(f"No TIFF files found in directory: {directory}")
        
        # Load all files into RAM
        images = {}
        for tiff_file in tqdm(tiff_files, desc="Loading files into RAM"):
            try:
                index = int(tiff_file.stem.split('_')[-1])
                images[index] = np.array(Image.open(tiff_file))
            except Exception as e:
                raise DataLoadError(f"Failed to load {tiff_file}: {str(e)}")
        
        return images
    except Exception as e:
        raise DataLoadError(f"Error loading TIFF files: {str(e)}")

def process_image(args):
    """Process a single image that's already in memory"""
    index, image = args
    return index, np.mean(image)

def analyze_tiff_sequence(images):
    """Process images that are already in RAM"""
    num_threads = cpu_count() // 2
    
    # Process images in parallel
    results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Create list of (index, image) tuples for processing
        items = list(images.items())
        
        # Process with progress bar
        list_of_results = list(tqdm(
            executor.map(process_image, items),
            total=len(items),
            desc="Processing images"
        ))
        
        results.extend(list_of_results)
    
    # Sort results by index
    results.sort(key=lambda x: x[0])
    
    return [r[0] for r in results], [r[1] for r in results]

def find_plateau_boundaries(values):
    # Calculate gradient
    gradient = np.gradient(values)
    abs_gradient = np.abs(gradient)
    
    # Use a more robust method to find transitions
    gradient_mean = np.mean(abs_gradient)
    gradient_std = np.std(abs_gradient)
    significant_change = gradient_mean + 3 * gradient_std
    
    # Find regions of rapid change
    transition_points = np.where(abs_gradient > significant_change)[0]
    
    # Group transition points that are close together
    groups = np.split(transition_points, np.where(np.diff(transition_points) > 20)[0] + 1)
    
    # If we don't find clear transitions (or find fewer than 2 groups)
    if len(groups) < 2:
        print("No clear transitions found - using full dataset range")
        start_idx = 0
        end_idx = len(values) - 1
    else:
        # First major transition is start, last major transition is end
        start_idx = groups[0][-1] + 10  # Add offset to get past transition
        end_idx = groups[-1][0] - 10    # Subtract offset to stop before transition
    
    analysis_data = {
        'threshold': significant_change,
        'gradients': abs_gradient,
        'transition_points': transition_points,
        'stability_scores': np.convolve(abs_gradient < significant_change, 
                                      np.ones(20)/20, mode='same'),
        'auto_range': len(groups) < 2  # Flag to indicate if we used auto-range
    }
    
    return start_idx, end_idx, analysis_data

def plot_threshold_diagnostics(analysis_data, save_path):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    plt.plot(analysis_data['gradients'], label='|Gradient|')
    plt.axhline(y=analysis_data['threshold'], color='r', linestyle=':', 
                label=f"Threshold: {analysis_data['threshold']:.2e}")
    plt.ylabel('|Gradient|')
    plt.legend()
    
    plt.subplot(212)
    plt.plot(analysis_data['stability_scores'], label='Stability Score')
    plt.axhline(y=0.8, color='r', linestyle=':', label='Stability Threshold')
    transition_points = analysis_data['transition_points']
    if len(transition_points) > 0:
        plt.plot(transition_points, 
                [0.5] * len(transition_points), 
                'r|', markersize=10, 
                label='Transition Points')
    plt.ylabel('Stability Score')
    plt.xlabel('Slice Number')
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()

def plot_results(slice_numbers, avg_values, start_idx, end_idx, threshold, save_path):
    plt.figure(figsize=(12, 7))
    plt.plot(slice_numbers, avg_values, '-b', linewidth=1)
    
    if start_idx is not None and end_idx is not None:
        plt.axvline(x=start_idx, color='r', linestyle=':', label=f'Start: {start_idx}')
        plt.axvline(x=end_idx, color='r', linestyle=':', label=f'End: {end_idx}')
        
        if start_idx == 0 and end_idx == len(avg_values) - 1:
            plt.text(0.02, 0.98, 'Using full dataset (no clear transitions found)', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    color='red')
        else:
            plt.text(0.02, 0.94, f'Detected threshold: {threshold:.2e}', 
                    transform=plt.gca().transAxes, verticalalignment='top')
            plt.text(0.02, 0.98, f'Plateau range: {end_idx - start_idx} slices', 
                    transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.xlabel('Slice Number')
    plt.ylabel('Average Pixel Value')
    plt.title('Average Pixel Value Across Image Sequence')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_xz_slice(images, start_idx, end_idx, save_path):
    """Create XZ slice visualization with cutoff lines"""
    sorted_indices = sorted(images.keys())
    volume = np.stack([images[i] for i in sorted_indices])
    
    mid_y = volume.shape[2] // 2
    xz_slice = volume[:, :, mid_y]
    
    plt.figure(figsize=(12, 8))
    plt.imshow(xz_slice, aspect='auto', cmap='gray')
    
    if start_idx is not None and end_idx is not None:
        plt.axhline(y=start_idx, color='r', linestyle=':', label=f'Start: {start_idx}')
        plt.axhline(y=end_idx, color='r', linestyle=':', label=f'End: {end_idx}')
        
        if start_idx == 0 and end_idx == len(sorted_indices) - 1:
            plt.text(0.02, 0.98, 'Using full dataset (no clear transitions found)', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    color='red')
    
    plt.xlabel('X position')
    plt.ylabel('Z position (slice number)')
    plt.title('XZ Slice Through Middle of Volume')
    plt.colorbar(label='Pixel Value')
    plt.legend()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# New functions
def setup_logging() -> logging.Logger:
    """Configure logging with file and console output"""
    logger = logging.getLogger('tomography_processor')
    logger.setLevel(logging.INFO)
    
    # Create formatters and handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler('tomography_processing.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def clip_and_rescale_volume(volume: np.ndarray) -> np.ndarray:
    """Clip volume to 1st/99th percentiles and rescale to uint8"""
    # Calculate percentiles
    p1, p99 = np.percentile(volume, [0.5, 99.5])
    
    # Clip the volume
    clipped = np.clip(volume, p1, p99)
    
    # Rescale to 0-255
    scaled = ((clipped - p1) / (p99 - p1) * 255).astype(np.uint8)
    
    return scaled

def save_volume(volume: np.ndarray, output_dir: Path, multipage: bool = True) -> bool:
    """Save the processed volume as a TIFF stack with error handling"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if multipage:
            try:
                output_path = output_dir / 'volume.tiff'
                tifffile.imwrite(output_path, volume)
                return True
            except Exception as e:
                logging.warning(f"Failed to save multipage TIFF: {e}. Falling back to single files.")
                multipage = False
        
        if not multipage:
            for i, slice_data in enumerate(volume):
                output_path = output_dir / f'slice_{i:04d}.tiff'
                Image.fromarray(slice_data).save(output_path)
        return True
    except Exception as e:
        raise ProcessingError(f"Failed to save volume: {str(e)}")

def process_single_experiment(
    exp_number: str, 
    input_base: Path,
    output_base: Path,
    logger: logging.Logger
) -> ProcessingResult:
    """Process a single experiment with comprehensive error handling"""
    import time
    start_time = time.time()
    
    try:
        # Find the Savu directory
        savu_dir = input_base / f'Savu_k11-{exp_number}_full_fd_rr_vo_pag_tiff'
        if not savu_dir.exists():
            raise DirectoryNotFoundError(
                f"Could not find Savu directory for experiment {exp_number}"
            )
            
        # Find TiffSaver directory
        tiffsaver_dirs = list(savu_dir.glob('TiffSaver_*'))
        if not tiffsaver_dirs:
            raise DirectoryNotFoundError(
                f"No TiffSaver directory found in {savu_dir}"
            )
        
        exp_input = tiffsaver_dirs[0]
        if len(tiffsaver_dirs) > 1:
            logger.warning(
                f"Multiple TiffSaver directories found in {savu_dir}, using {exp_input}"
            )
        
        exp_output = output_base / exp_number
        exp_output.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing experiment {exp_number}")
        logger.info(f"Using TiffSaver directory: {exp_input}")
        
        # Load and process images with specific error handling for each step
        try:
            images = load_tiffs_to_ram(exp_input)
        except DataLoadError as e:
            return ProcessingResult(
                exp_number,
                ProcessingStatus.DATA_LOAD_FAILED,
                str(e),
                time.time() - start_time
            )
        
        try:
            # Analysis steps
            slice_numbers, avg_values = analyze_tiff_sequence(images)
            start_idx, end_idx, analysis_data = find_plateau_boundaries(avg_values)
            
            # Save diagnostic plots
            plot_threshold_diagnostics(
                analysis_data, 
                exp_output / f'{exp_number}_threshold_analysis.png'
            )
            plot_results(
                slice_numbers, avg_values, start_idx, end_idx,
                analysis_data['threshold'], 
                exp_output / f'{exp_number}_average_pixel_values.png'
            )
            plot_xz_slice(
                images, start_idx, end_idx,
                exp_output / f'{exp_number}_xz_slice.png'
            )
            
            # Process volume
            sorted_indices = sorted(images.keys())
            volume = np.stack([images[i] for i in sorted_indices])
            trimmed_volume = volume[start_idx:end_idx]
            processed_volume = clip_and_rescale_volume(trimmed_volume)
            
        except Exception as e:
            return ProcessingResult(
                exp_number,
                ProcessingStatus.PROCESSING_FAILED,
                str(e),
                time.time() - start_time
            )
        
        try:
            # Save processed volume
            save_volume(processed_volume, exp_output / 'trimmed_data')
        except ProcessingError as e:
            return ProcessingResult(
                exp_number,
                ProcessingStatus.SAVE_FAILED,
                str(e),
                time.time() - start_time
            )
        
        return ProcessingResult(
            exp_number,
            ProcessingStatus.SUCCESS,
            None,
            time.time() - start_time
        )
        
    except DirectoryNotFoundError as e:
        return ProcessingResult(
            exp_number,
            ProcessingStatus.DIRECTORY_NOT_FOUND,
            str(e),
            time.time() - start_time
        )
    except Exception as e:
        return ProcessingResult(
            exp_number,
            ProcessingStatus.PROCESSING_FAILED,
            str(e),
            time.time() - start_time
        )

def generate_summary_report(results: List[ProcessingResult], output_path: Path):
    """Generate a detailed summary report of all processing results"""
    with open(output_path / 'processing_summary.txt', 'w') as f:
        f.write("Processing Summary Report\n")
        f.write("=======================\n\n")
        
        # Overall statistics
        total = len(results)
        successful = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS)
        f.write(f"Total experiments processed: {total}\n")
        f.write(f"Successfully processed: {successful}\n")
        f.write(f"Failed: {total - successful}\n\n")
        
        # Detailed results
        f.write("Detailed Results:\n")
        f.write("-----------------\n")
        for result in results:
            f.write(f"\nExperiment: {result.experiment_number}\n")
            f.write(f"Status: {result.status.name}\n")
            f.write(f"Processing time: {result.processing_time:.2f} seconds\n")
            if result.error_message:
                f.write(f"Error: {result.error_message}\n")

def main():
    # Setup logging
    logger = setup_logging()
    
    # Base paths
    input_base = Path('/dls/k11/data/2024/mg40414-1/processed')
    output_base = Path('/dls/science/users/qps56811/data/mg40414-1')
    
    # List of experiment numbers to process
    experiment_numbers = [
        '52548',
        '52549',
        '52551',
        '52553',
        '52555',
        '52557',
        '52559',
        '52561',
        '52565',
        '52567',
        '52571',
        '52573',
        '52575',
        '52577',
        '52579',
        '52581',
        '52583',
        '52585',
        '52587',
        '52589',
        '52592',
        '52595',
        '52605',
        '52607',
        '52609',
        '52611',
        '52613',
        '52615',
        '52617',
        '52619',
        '52621',
        '52623',
        '52625'
    ]
    
    # Process each experiment
    results = []
    for exp_num in experiment_numbers:
        try:
            result = process_single_experiment(
                exp_num, 
                input_base, 
                output_base,
                logger
            )
            results.append(result)
            
            # Log the result
            if result.status == ProcessingStatus.SUCCESS:
                logger.info(
                    f"Successfully processed experiment {exp_num} "
                    f"in {result.processing_time:.2f} seconds"
                )
            else:
                logger.error(
                    f"Failed to process experiment {exp_num}: "
                    f"{result.status.name} - {result.error_message}"
                )
                
        except Exception as e:
            logger.error(f"Unexpected error processing experiment {exp_num}: {str(e)}")
            results.append(ProcessingResult(
                exp_num,
                ProcessingStatus.PROCESSING_FAILED,
                str(e)
            ))
    
    # Generate summary report
    try:
        generate_summary_report(results, output_base)
        logger.info(f"Summary report generated at {output_base}/processing_summary.txt")
    except Exception as e:
        logger.error(f"Failed to generate summary report: {str(e)}")
    
    # Print final summary to console
    logger.info("\nProcessing Complete")
    logger.info("==================")
    successful = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS)
    logger.info(f"Successfully processed: {successful}/{len(results)} experiments")

if __name__ == "__main__":
    main()
