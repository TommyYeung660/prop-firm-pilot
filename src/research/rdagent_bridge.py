"""
Bridge to qlib_rd_agent — triggers weekend factor research
and manages the discovered_factors.yaml lifecycle.

RD-Agent runs offline (weekends) to discover new FX alpha factors,
which are then injected into the scanner's Alpha158 feature pipeline
via the existing load_rdagent_factors() interface.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
from loguru import logger


class DiscoveredFactor:
    """A single factor discovered by RD-Agent."""

    def __init__(
        self,
        name: str,
        expression: str,
        ic_mean: float = 0.0,
        ic_ir: float = 0.0,
        description: str = "",
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.expression = expression
        self.ic_mean = ic_mean
        self.ic_ir = ic_ir
        self.description = description
        self.enabled = enabled

    def __repr__(self) -> str:
        return f"Factor({self.name}, IC_IR={self.ic_ir:.2f}, enabled={self.enabled})"


class RdAgentBridge:
    """Bridge to qlib_rd_agent for weekend factor research.

    Workflow:
    1. Trigger RD-Agent to sync FX data from Dropbox
    2. Run factor mining (RD-Agent's AI loop)
    3. Upload discovered_factors.yaml to Dropbox
    4. Scanner's load_rdagent_factors() picks up new factors automatically

    Usage:
        bridge = RdAgentBridge(rdagent_path="../../qlib_rd_agent")
        bridge.trigger_full_cycle()
        factors = bridge.load_factors(min_ic_ir=0.5)
    """

    def __init__(
        self,
        rdagent_path: str | Path,
        factors_yaml_path: str | Path | None = None,
    ) -> None:
        self._rdagent_path = Path(rdagent_path).resolve()
        self._factors_yaml_path = (
            Path(factors_yaml_path)
            if factors_yaml_path
            else self._rdagent_path / "output" / "discovered_factors.yaml"
        )

    def trigger_full_cycle(self, timeout: int = 3600) -> bool:
        """Run the full RD-Agent cycle: sync → run → upload.

        Args:
            timeout: Maximum execution time in seconds (default 1h).

        Returns:
            True if cycle completed successfully.
        """
        logger.info("RdAgentBridge: triggering full cycle (timeout={}s)", timeout)

        cmd = [sys.executable, "-m", "src.main", "full"]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._rdagent_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                logger.error(
                    "RdAgentBridge: full cycle failed (exit={}):\n{}",
                    result.returncode,
                    result.stderr[:500],
                )
                return False

            logger.info("RdAgentBridge: full cycle completed successfully")
            return True

        except subprocess.TimeoutExpired:
            logger.error("RdAgentBridge: full cycle timed out after {}s", timeout)
            return False
        except FileNotFoundError as e:
            logger.error("RdAgentBridge: failed to run rd_agent: {}", e)
            return False

    def trigger_sync(self) -> bool:
        """Sync data from Dropbox (download shared FX data)."""
        return self._run_command("sync")

    def trigger_upload(self) -> bool:
        """Upload discovered factors to Dropbox."""
        return self._run_command("upload")

    def load_factors(self, min_ic_ir: float = 0.5) -> List[DiscoveredFactor]:
        """Load discovered factors from YAML, filtered by IC IR threshold.

        Args:
            min_ic_ir: Minimum IC IR to include (default 0.5).

        Returns:
            List of factors that pass the threshold.
        """
        if not self._factors_yaml_path.exists():
            logger.warning("RdAgentBridge: factors YAML not found: {}", self._factors_yaml_path)
            return []

        with open(self._factors_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "factors" not in data:
            logger.warning("RdAgentBridge: no factors in YAML")
            return []

        all_factors = []
        for item in data["factors"]:
            factor = DiscoveredFactor(
                name=item.get("name", ""),
                expression=item.get("expression", ""),
                ic_mean=float(item.get("ic_mean", 0)),
                ic_ir=float(item.get("ic_ir", 0)),
                description=item.get("description", ""),
                enabled=item.get("enabled", True),
            )
            all_factors.append(factor)

        filtered = [f for f in all_factors if f.enabled and f.ic_ir >= min_ic_ir]

        logger.info(
            "RdAgentBridge: loaded {}/{} factors (min_ic_ir={})",
            len(filtered),
            len(all_factors),
            min_ic_ir,
        )
        return filtered

    def _run_command(self, command: str, timeout: int = 300) -> bool:
        """Run a single RD-Agent CLI command."""
        cmd = [sys.executable, "-m", "src.main", command]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self._rdagent_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0:
                logger.error(
                    "RdAgentBridge: '{}' failed (exit={}): {}",
                    command,
                    result.returncode,
                    result.stderr[:300],
                )
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error("RdAgentBridge: '{}' timed out", command)
            return False
        except FileNotFoundError as e:
            logger.error("RdAgentBridge: command failed: {}", e)
            return False
