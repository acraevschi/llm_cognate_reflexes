"""Download Lexibank CLDF datasets and Glottolog via shallow git clones.

This module provides :class:`LexibankDownloader`, which uses the GitHub
REST API (via :mod:`urllib.request` — no ``requests`` dependency) to
enumerate repositories in the ``lexibank`` organisation and shallow-clone
them for local CLDF processing.
"""

from __future__ import annotations

import json
import logging
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"
_LEXIBANK_ORG = "lexibank"
_GLOTTOLOG_CLONE_URL = "https://github.com/glottolog/glottolog.git"

# GitHub API returns at most 100 items per page.
_PER_PAGE = 100


class LexibankDownloader:
    """Clone Lexibank dataset repositories from GitHub.

    Each dataset is cloned with ``git clone --depth 1`` (shallow) into
    *data_dir*.  Already-cloned directories are silently skipped.

    Args:
        data_dir: Local directory for storing cloned repos.
    """

    def __init__(self, data_dir: str | Path = "./data/lexibank") -> None:
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_all(self, org: str = _LEXIBANK_ORG) -> list[Path]:
        """Clone every repository from a GitHub organisation.

        Uses the GitHub REST API to paginate through all repositories,
        then shallow-clones each one that has not already been downloaded.

        Args:
            org: GitHub organisation name (default ``"lexibank"``).

        Returns:
            A list of :class:`~pathlib.Path` objects pointing to the
            cloned (or already-existing) repository directories.
        """
        repos = self._list_org_repos(org)
        logger.info("Found %d repositories in '%s'.", len(repos), org)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        for name, clone_url in repos:
            dest = self.data_dir / name
            if dest.exists():
                logger.debug("Skipping '%s' — already cloned.", name)
                paths.append(dest)
                continue
            self._git_clone(clone_url, dest)
            paths.append(dest)

        return paths

    def download_dataset(self, name: str, org: str = _LEXIBANK_ORG) -> Path:
        """Clone a single dataset repository by name.

        Args:
            name: Repository name (e.g. ``"abvd"``).
            org: GitHub organisation (default ``"lexibank"``).

        Returns:
            Path to the cloned repository directory.
        """
        dest = self.data_dir / name
        if dest.exists():
            logger.info("Dataset '%s' already exists at %s.", name, dest)
            return dest

        clone_url = f"https://github.com/{org}/{name}.git"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._git_clone(clone_url, dest)
        return dest

    def download_glottolog(self, target_dir: str | Path = "./data/glottolog") -> Path:
        """Clone the Glottolog repository.

        Args:
            target_dir: Where to place the clone.

        Returns:
            Path to the cloned Glottolog directory.
        """
        dest = Path(target_dir)
        if dest.exists():
            logger.info("Glottolog already exists at %s.", dest)
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        self._git_clone(_GLOTTOLOG_CLONE_URL, dest)
        return dest

    def list_downloaded(self) -> list[str]:
        """Return names of already-downloaded datasets.

        Only directories that look like git repos (contain a ``.git``
        sub-directory) are included.

        Returns:
            Sorted list of dataset directory names.
        """
        if not self.data_dir.exists():
            return []

        return sorted(
            d.name
            for d in self.data_dir.iterdir()
            if d.is_dir() and (d / ".git").exists()
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _list_org_repos(org: str) -> list[tuple[str, str]]:
        """Fetch all repositories for a GitHub organisation.

        Paginates through the GitHub REST API (up to *_PER_PAGE* items per
        request) and returns ``(name, clone_url)`` pairs.

        Args:
            org: GitHub organisation name.

        Returns:
            List of ``(repo_name, clone_url)`` tuples.

        Raises:
            urllib.error.URLError: On network / API errors.
        """
        repos: list[tuple[str, str]] = []
        page = 1

        while True:
            url = (
                f"{_GITHUB_API_BASE}/orgs/{org}/repos"
                f"?per_page={_PER_PAGE}&page={page}&type=public"
            )
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "cognate_reflexes/0.1",
                },
            )
            try:
                with urllib.request.urlopen(request) as response:  # noqa: S310
                    data: list[dict] = json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                logger.error(
                    "GitHub API request failed (page %d): %s", page, exc
                )
                raise

            if not data:
                break

            for repo in data:
                repos.append((repo["name"], repo["clone_url"]))

            # Fewer results than a full page means we've reached the end.
            if len(data) < _PER_PAGE:
                break

            page += 1

        return repos

    @staticmethod
    def _git_clone(clone_url: str, dest: Path) -> None:
        """Perform a shallow git clone.

        Args:
            clone_url: HTTPS clone URL.
            dest: Local destination directory.

        Raises:
            subprocess.CalledProcessError: If git exits with non-zero status.
        """
        logger.info("Cloning %s -> %s", clone_url, dest)
        subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Cloned %s successfully.", dest.name)
