import os
import sys
import platform
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class HardwareDetector:
    """
    Detects supported hardware for JAX, CuPy (CUDA/HIP), DPC++ (OneAPI), and PyOpenCL
    without importing the actual computation packages.
    """
    
    def __init__(self):
        self.system = platform.system().lower()
        self.arch = platform.machine().lower()
        self.results = {}
    
    def detect_all(self) -> Dict[str, Dict]:
        """
        Detect all supported hardware frameworks.
        Returns a dictionary with detection results.
        """
        self.results = {
            'jax': self.detect_jax_support(),
            'cupy_cuda': self.detect_cuda_support(),
            'cupy_hip': self.detect_hip_support(),
            'dpcpp': self.detect_oneapi_support(),
            'opencl': self.detect_opencl_support()
        }
        return self.results
    
    def detect_jax_support(self) -> Dict:
        """
        Detect JAX support by checking for available backends.
        JAX supports CPU, GPU (CUDA), and TPU.
        """
        result = {
            'supported': False,
            'backends': [],
            'gpu_available': False,
            'tpu_available': False,
            'details': {}
        }
        
        # Check CPU backend (always available)
        result['backends'].append('cpu')
        
        # Check CUDA support
        cuda_info = self.detect_cuda_support()
        if cuda_info['supported']:
            result['backends'].append('gpu')
            result['gpu_available'] = True
            result['details']['cuda'] = cuda_info
        
        # Check TPU support (simplified check)
        tpu_available = self._check_tpu_support()
        if tpu_available:
            result['backends'].append('tpu')
            result['tpu_available'] = True
        
        result['supported'] = len(result['backends']) > 1  # More than just CPU
        
        return result
    
    def detect_cuda_support(self) -> Dict:
        """
        Detect CUDA support for CuPy.
               """
        result = {
            'supported': False,
            'cuda_available': False,
            'cuda_version': None,
            'gpu_count': 0,
            'gpu_info': [],
            'nvcc_available': False
        }
        
        # Check for nvidia-smi
        nvidia_smi_path = self._find_nvidia_smi()
        if nvidia_smi_path:
            try:
                # Get GPU info using nvidia-smi
                output = subprocess.check_output([
                    nvidia_smi_path, 
                    '--query-gpu=name,driver_version,memory.total',
                    '--format=csv,noheader,nounits'
                ], text=True, stderr=subprocess.DEVNULL)
                
                gpus = output.strip().split('\n')
                result['gpu_count'] = len(gpus)
                result['gpu_info'] = gpus
                result['cuda_available'] = True
                
                # Try to get CUDA version from nvidia-smi
                try:
                    cuda_output = subprocess.check_output([
                        nvidia_smi_path, '--query', '--display=DRIVER,CUDA',
                        '--format=csv,noheader'
                    ], text=True, stderr=subprocess.DEVNULL)
                    if 'CUDA Version' in cuda_output:
                        match = re.search(r'CUDA Version:\s*(\d+\.\d+)', cuda_output)
                        if match:
                            result['cuda_version'] = match.group(1)
                except:
                    pass
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        # Check for CUDA toolkit installation
        cuda_paths = [
            os.environ.get('CUDA_PATH'),
            os.environ.get('CUDA_HOME'),
            '/usr/local/cuda',
            'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA'
        ]
        
        cuda_toolkit_found = False
        for path in cuda_paths:
            if path and Path(path).exists():
                cuda_toolkit_found = True
                # Check for nvcc
                nvcc_path = Path(path) / 'bin' / 'nvcc'
                if nvcc_path.exists():
                    result['nvnvcc_available'] = True
                    if not result['cuda_version']:
                        # Try to get version from nvcc
                        try:
                            nvcc_output = subprocess.check_output(
                                [str(nvcc_path), '--version'], 
                                text=True, stderr=subprocess.DEVNULL
                            )
                            match = re.search(r'release\s+(\d+\.\d+)', nvcc_output)
                            if match:
                                result['cuda_version'] = match.group(1)
                        except:
                            pass
                break
        
        result['supported'] = result['cuda_available'] and cuda_toolkit_found
        
        return result
    
    def detect_hip_support(self) -> Dict:
        """
        Detect HIP/ROCm support for AMD GPUs.
        """
        result = {
            'supported': False,
            'hip_available': False,
            'rocm_available': False,
            'rocm_version': None,
            'amd_gpu_count': 0,
            'gpu_info': []
        }
        
        # Check for ROCm installation
        rocm_paths = [
            os.environ.get('ROCM_PATH'),
            '/opt/rocm',
            '/usr/local/rocm'
        ]
        
        rocm_found = False
        for path in rocm_paths:
            if path and Path(path).exists():
                rocm_found = True
                # Try to get ROCm version
                version_file = Path(path) / '.info' / 'version'
                if version_file.exists():
                    try:
                        result['rocm_version'] = version_file.read_text().strip()
                    except:
                        pass
                break
        
        # Check for AMD GPUs (simplified check)
        if self.system == 'linux':
            # Check for AMD GPU devices
            amd_gpu_dirs = list(Path('/sys/class/drm').glob('card*/device/vendor'))
            amd_gpus = []
            for vendor_file in amd_gpu_dirs:
                try:
                    vendor_id = vendor_file.read_text().strip()
                    if vendor_id == '0x1002':  # AMD vendor ID
                        gpu_path = vendor_file.parent
                        gpu_name = (gpu_path / 'name').read_text().strip() if (gpu_path / 'name').exists() else 'AMD GPU'
                        amd_gpus.append(gpu_name)
                except:
                    continue
            
            result['amd_gpu_count'] = len(amd_gpus)
            result['gpu_info'] = amd_gpus
        
        result['hip_available'] = rocm_found
        result['rocm_available'] = rocm_found
        result['supported'] = rocm_found and result['amd_gpu_count'] > 0
        
        return result
    
    def detect_oneapi_support(self) -> Dict:
        """
        Detect OneAPI/DPC++ support.
        """
        result = {
            'supported': False,
            'oneapi_available': False,
            'intel_gpu_available': False,
            'intel_cpu_available': False,
            'compiler_available': False,
            'details': {}
        }
        
        # Check for OneAPI environment variables
        oneapi_vars = [
            'ONEAPI_ROOT', 'INTEL_ONEAPI_ROOT', 
            'CMPLR_ROOT', 'DPCPP_ROOT'
        ]
        
        oneapi_found = False
        for var in oneapi_vars:
            if os.environ.get(var):
                oneapi_found = True
                result['details']['env_var'] = var
                break
        
        # Check for DPC++ compiler
        compilers = ['dpcpp', 'icpx', 'icx']
        compiler_found = False
        for compiler in compilers:
            try:
                subprocess.run([compiler, '--version'], 
                            capture_output=True, check=False)
                compiler_found = True
                result['details']['compiler'] = compiler
                break
            except FileNotFoundError:
                continue
        
        # Check for Intel GPU (simplified)
        if self.system == 'linux':
            intel_gpu_dirs = list(Path('/sys/class/drm').glob('card*/device/vendor'))
            intel_gpus = 0
            for vendor_file in intel_gpu_dirs:
                try:
                    vendor_id = vendor_file.read_text().strip()
                    if vendor_id == '0x8086':  # Intel vendor ID
                        intel_gpus += 1
                except:
                    continue
            result['intel_gpu_available'] = intel_gpus > 0
        
        # Intel CPU detection
        cpu_info = self._get_cpu_info()
        result['intel_cpu_available'] = 'intel' in cpu_info.get('vendor_id', '').lower()
        
        result['oneapi_available'] = oneapi_found
        result['compiler_available'] = compiler_found
        result['supported'] = (oneapi_found or compiler_found) and (
            result['intel_gpu_available'] or result['intel_cpu_available']
        )
        
        return result
    
    def detect_opencl_support(self) -> Dict:
        """
        Detect OpenCL support.
        """
        result = {
            'supported': False,
            'platforms_available': 0,
            'devices_available': 0,
            'platform_info': []
        }
        
        # Check for OpenCL ICD files (Linux)
        if self.system == 'linux':
            icd_paths = [
                '/etc/OpenCL/vendors',
                '/usr/local/etc/OpenCL/vendors'
            ]
            for icd_path in icd_paths:
                if Path(icd_path).exists():
                    result['platforms_available'] += len(list(Path(icd_path).glob('*.icd')))
        
        # Check for OpenCL registry entries (Windows)
        elif self.system == 'windows':
            try:
                import winreg
                key_paths = [
                    r"SOFTWARE\Khronos\OpenCL\Vendors",
                    r"SOFTWARE\Wow6432Node\Khronos\OpenCL\Vendors"
                ]
                for key_path in key_paths:
                    try:
                        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                            result['platforms_available'] += 1
                    except FileNotFoundError:
                        continue
            except ImportError:
                pass
        
        # Check for common OpenCL library files
        opencl_libs = ['libOpenCL.so', 'OpenCL.dll', 'OpenCL.framework']
        for lib in opencl_libs:
            if self._find_library(lib):
                result['devices_available'] = 1  # Simplified
        
        result['supported'] = result['platforms_available'] > 0 or result['devices_available'] > 0
        
        return result
    
    def _find_nvidia_smi(self) -> Optional[str]:
        """Find nvidia-smi executable path."""
        if self.system == 'windows':
            paths = [
                os.environ.get('CUDA_PATH', ''),
                'C:\\Program Files\\NVIDIA Corporation\\NVSMI',
                'C:\\Windows\\System32'
            ]
            exe_name = 'nvidia-smi.exe'
        else:
            paths = ['/usr/bin', '/usr/local/bin', '/opt/cuda/bin']
            exe_name = 'nvidia-smi'
        
        for path in paths:
            full_path = Path(path) / exe_name
            if full_path.exists():
                return str(full_path)
        
        # Check PATH
        try:
            subprocess.run([exe_name, '--version'], capture_output=True, check=False)
            return exe_name
        except FileNotFoundError:
            return None
    
    def _find_library(self, lib_name: str) -> bool:
        """Check if a library exists in common paths."""
        lib_paths = []
        if self.system == 'linux':
            lib_paths = ['/usr/lib', '/usr/lib64', '/usr/local/lib']
        elif self.system == 'windows':
            lib_paths = [
                os.environ.get('SystemRoot', '') + '\\System32',
                os.environ.get('ProgramFiles', ''),
            ]
        elif self.system == 'darwin':
            lib_paths = ['/usr/lib', '/usr/local/lib', '/Library/Frameworks']
        
        for path in lib_paths:
            if (Path(path) / lib_name).exists():
                return True
        return False
    
    def _check_tpu_support(self) -> bool:
        """Simplified TPU support check."""
        # Check for Google TPU environment variables
        tpu_vars = ['TPU_NAME', 'TPU_IP_ADDRESS', 'COLAB_TPU_ADDR']
        for var in tpu_vars:
            if os.environ.get(var):
                return True
        
        # Check if running on Google Colab
        try:
            import google.colab
            return True
        except ImportError:
            pass
        
        return False
    
    def _get_cpu_info(self) -> Dict:
        """Get basic CPU information."""
        cpu_info = {}
        try:
            if self.system == 'linux':
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'vendor_id' in line:
                            cpu_info['vendor_id'] = line.split(':')[1].strip()
                        elif 'model name' in line:
                            cpu_info['model'] = line.split(':')[1].strip()
                            break
            elif self.system == 'windows':
                output = subprocess.check_output(
                    ['wmic', 'cpu', 'get', 'name,manufacturer'], 
                    text=True
                )
                lines = output.strip().split('\n')
                if len(lines) > 1:
                    cpu_info['model'] = lines[1].strip()
        except:
            pass
        
        return cpu_info
    
    def print_summary(self):
        """Print a summary of detection results."""
        print("Hardwareware Support Detection Summary")
        print("=" * 50)
        
        for framework, info in self.results.items():
            status = "✓ SUPPORTED" if info.get('supported', False) else "✗ NOT SUPPORTED"
            print(f"{framework.upper():<12}: {status}")
            
            if info.get('supported', False):
                # Print additional details for supported frameworks
                if framework == 'jax':
                    backends = info.get('backends', [])
                    print(f"    Backends: {', '.join(backends)}")
                elif framework == 'cupy_cuda':
                    print(f"    CUDA Version: {info.get('cuda_version', 'Unknown')}")
                    print(f"    GPUs: {info.get('gpu_count', 0)}")
                elif framework == 'cupy_hip':
                    print(f"    ROCm Version: {info.get('rocm_version', 'Unknown')}")
                    print(f"    AMD GPUs: {info.get('amd_gpu_count', 0)}")
                elif framework == 'dpcpp':
                    compilers = ['dpcpp', 'icpx', 'icx']
                    available = any(self._check_compiler(c) for c in compilers)
                    print(f"    Compiler: {'Available' if available else 'Not found'}")
                elif framework == 'opencl':
                    print(f"    Platforms: {info.get('platforms_available', 0)}")
            
            print()

    def _check_compiler(self, compiler: str) -> bool:
        """Check if a compiler is available."""
        try:
            subprocess.run([compiler, '--version'], 
                         capture_output=True, check=False)
            return True
        except FileNotFoundError:
            return False

# Example usage
if __name__ == "__main__":
    detector = HardwareDetector()
    results = detector.detect_all()
    detector.print_summary()
    
    # You can also access specific results
    print("\nDetailed Results:")
    for framework, info in results.items():
        print(f"\n{framework}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

