"""
Docker Lifecycle Management for Memvid MCP Server

Handles Docker installation, container startup/shutdown for memvid MCP server.
"""

import asyncio
import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger(__name__)


class DockerLifecycleManager:
    """Manages Docker and memvid container lifecycle."""

    def __init__(self):
        self.container_name = "memvid-h265"
        self.container_id: Optional[str] = None
        self.docker_cmd = self._find_docker_command()

    def _find_docker_command(self) -> Optional[str]:
        """Find appropriate Docker command (WSL compatible)"""
        if shutil.which("docker.exe"):
            return "docker.exe"
        elif shutil.which("docker"):
            return "docker"
        return None

    async def ensure_docker_installed(self) -> bool:
        """Ensure Docker is installed, install if needed."""
        try:
            if self.docker_cmd:
                proc = await asyncio.create_subprocess_exec(
                    self.docker_cmd, "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await asyncio.wait_for(proc.wait(), timeout=5)
                if proc.returncode == 0:
                    logger.info(f"✅ Docker already installed: {self.docker_cmd}")
                    return True

            logger.info("🔧 Installing Docker...")

            current_user = os.getenv('USER') or os.getenv('USERNAME') or 'ubuntu'
            install_commands = [
                "sudo apt-get update",
                "sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release",
                "curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg",
                'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null',
                "sudo apt-get update",
                "sudo apt-get install -y docker-ce docker-ce-cli containerd.io",
                "sudo systemctl start docker",
                "sudo systemctl enable docker",
                f"sudo usermod -aG docker {current_user}"
            ]

            for cmd in install_commands:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await proc.wait()
                if proc.returncode != 0:
                    logger.error(f"Command failed: {cmd}")
                    return False

            self.docker_cmd = self._find_docker_command()
            return self.docker_cmd is not None

        except Exception as e:
            logger.error(f"Docker installation failed: {e}")
            return False

    async def start_container(self) -> bool:
        """Start the memvid container."""
        try:
            if not self.docker_cmd:
                return False

            # Check if already running
            proc = await asyncio.create_subprocess_exec(
                self.docker_cmd, "ps", "-q", "-f", f"ancestor={self.container_name}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()

            if proc.returncode == 0 and stdout.strip():
                self.container_id = stdout.decode().strip()
                logger.info(f"✅ Container already running: {self.container_id}")
                return True

            # Start container
            start_cmd = [
                self.docker_cmd, "run", "-d",
                "--name", f"{self.container_name}-session",
                self.container_name,
                "sleep", "infinity"
            ]

            proc = await asyncio.create_subprocess_exec(
                *start_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode == 0:
                self.container_id = stdout.decode().strip()
                logger.info(f"✅ Container started: {self.container_id}")
                return True
            else:
                logger.error(f"Failed to start container: {stderr.decode() if stderr else 'Unknown error'}")
                return False

        except Exception as e:
            logger.error(f"Container start error: {e}")
            return False

    async def stop_container(self) -> None:
        """Stop and remove the container."""
        try:
            if not self.docker_cmd or not self.container_id:
                return

            # Stop container
            proc = await asyncio.create_subprocess_exec(
                self.docker_cmd, "stop", f"{self.container_name}-session",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(proc.wait(), timeout=30)

            # Remove container
            proc = await asyncio.create_subprocess_exec(
                self.docker_cmd, "rm", f"{self.container_name}-session",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await asyncio.wait_for(proc.wait(), timeout=30)

            self.container_id = None
            logger.info("✅ Container stopped")

        except Exception as e:
            logger.error(f"Container stop error: {e}")

    async def initialize(self) -> bool:
        """Initialize Docker and start container."""
        if not await self.ensure_docker_installed():
            return False

        return await self.start_container()

    async def cleanup(self) -> None:
        """Stop container."""
        await self.stop_container()

    def get_status(self) -> dict:
        """Get Docker status."""
        return {
            "docker_available": self.docker_cmd is not None,
            "container_running": self.container_id is not None,
            "container_id": self.container_id
        }
