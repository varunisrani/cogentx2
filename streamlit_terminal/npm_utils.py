import os
import subprocess
import logging
import shlex
import platform
from pathlib import Path

def get_nvm_path():
    """Get the path to NVM installation"""
    home = str(Path.home())
    
    # Common NVM installation paths
    if platform.system() == "Windows":
        return os.path.join(home, ".nvm")
    else:  # macOS or Linux
        nvm_paths = [
            os.path.join(home, ".nvm"),
            os.path.join(home, ".config", "nvm"),
            "/usr/local/opt/nvm",  # Homebrew installation on macOS
        ]
        
        for path in nvm_paths:
            if os.path.exists(path):
                return path
    
    return None

def get_nvm_sh_path():
    """Get the path to nvm.sh script"""
    nvm_path = get_nvm_path()
    if not nvm_path:
        return None
    
    if platform.system() == "Windows":
        return os.path.join(nvm_path, "nvm.ps1")
    else:
        return os.path.join(nvm_path, "nvm.sh")

def is_nvm_installed():
    """Check if NVM is installed"""
    return get_nvm_sh_path() is not None

def is_npm_installed():
    """Check if npm is installed globally"""
    try:
        subprocess.run(
            ["npm", "--version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def get_node_versions():
    """Get list of installed Node.js versions via NVM"""
    if not is_nvm_installed():
        return []
    
    try:
        if platform.system() == "Windows":
            cmd = ["powershell", "-Command", f". '{get_nvm_sh_path()}'; nvm list"]
        else:
            cmd = ["bash", "-c", f"source '{get_nvm_sh_path()}' && nvm ls"]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        
        versions = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("v"):
                versions.append(line.split()[0])
            elif "->" in line and "v" in line:
                # Extract version from lines like "-> v16.14.0"
                parts = line.split("->")
                if len(parts) > 1:
                    v_part = parts[1].strip()
                    if v_part.startswith("v"):
                        versions.append(v_part.split()[0])
        
        return versions
    except subprocess.SubprocessError as e:
        logging.error(f"Error getting Node.js versions: {e}")
        return []

def get_npm_packages(global_packages=False):
    """Get list of installed npm packages"""
    if not is_npm_installed():
        return []
    
    try:
        cmd = ["npm", "list", "--depth=0"]
        if global_packages:
            cmd.append("-g")
            
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        packages = []
        for line in result.stdout.splitlines()[1:]:  # Skip the first line which is the project name
            if line.strip():
                packages.append(line.strip())
        
        return packages
    except subprocess.SubprocessError as e:
        logging.error(f"Error getting npm packages: {e}")
        return []

def get_nvm_env():
    """Get environment variables needed for NVM"""
    env = os.environ.copy()
    nvm_path = get_nvm_path()
    
    if nvm_path:
        if platform.system() == "Windows":
            # Windows uses different paths
            env["NVM_HOME"] = nvm_path
            env["NVM_SYMLINK"] = os.path.join(nvm_path, "current")
            if "PATH" in env:
                env["PATH"] = f"{os.path.join(nvm_path, 'current')};{env['PATH']}"
        else:
            # Unix-like systems
            env["NVM_DIR"] = nvm_path
            if "PATH" in env:
                env["PATH"] = f"{os.path.join(nvm_path, 'current', 'bin')}:{env['PATH']}"
    
    return env

def create_nvm_command(command):
    """Create a command that uses NVM to execute npm/node commands"""
    if not is_nvm_installed():
        return command
    
    nvm_sh = get_nvm_sh_path()
    if not nvm_sh:
        return command
    
    if platform.system() == "Windows":
        return f"powershell -Command \". '{nvm_sh}'; {command}\""
    else:
        return f"source '{nvm_sh}' && {command}"
