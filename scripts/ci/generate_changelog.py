#!/usr/bin/env python3
"""Generate changelog from git commits since last tag."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_last_tag() -> str:
    """Get the most recent git tag."""
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # No tags yet
        return ""


def get_commits_since_tag(tag: str) -> list[dict[str, str]]:
    """Get all commits since the given tag."""
    if tag:
        git_range = f"{tag}..HEAD"
    else:
        git_range = "HEAD"

    try:
        result = subprocess.run(
            [
                "git",
                "log",
                git_range,
                "--pretty=format:%H|%s|%an|%ae|%ad",
                "--date=short",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|")
            if len(parts) == 5:
                commits.append(
                    {
                        "hash": parts[0][:7],
                        "subject": parts[1],
                        "author": parts[2],
                        "email": parts[3],
                        "date": parts[4],
                    },
                )

        return commits
    except subprocess.CalledProcessError:
        return []


def categorize_commits(
    commits: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    """Categorize commits by type."""
    categories = {
        "features": [],
        "fixes": [],
        "docs": [],
        "tests": [],
        "ci": [],
        "refactor": [],
        "other": [],
    }

    for commit in commits:
        subject = commit["subject"].lower()

        if subject.startswith("feat") or "feature" in subject:
            categories["features"].append(commit)
        elif subject.startswith("fix") or "bug" in subject:
            categories["fixes"].append(commit)
        elif subject.startswith("docs") or "documentation" in subject:
            categories["docs"].append(commit)
        elif subject.startswith("test") or "tests" in subject:
            categories["tests"].append(commit)
        elif subject.startswith("ci") or "workflow" in subject or "github actions" in subject:
            categories["ci"].append(commit)
        elif subject.startswith("refactor") or "cleanup" in subject:
            categories["refactor"].append(commit)
        else:
            categories["other"].append(commit)

    return categories


def generate_changelog(version: str | None = None) -> str:
    """Generate a markdown changelog."""
    last_tag = get_last_tag()
    commits = get_commits_since_tag(last_tag)

    if not commits:
        return "No changes since last release."

    # Categorize commits
    categorized = categorize_commits(commits)

    # Build changelog
    changelog = []

    if version:
        changelog.append(f"# {version}\n")
    else:
        changelog.append("# Changelog\n")

    changelog.append(f"**Release Date:** {datetime.now().strftime('%Y-%m-%d')}\n")

    if last_tag:
        changelog.append(f"**Changes since:** {last_tag}\n")

    # Features
    if categorized["features"]:
        changelog.append("## ✨ Features\n")
        for commit in categorized["features"]:
            changelog.append(f"- {commit['subject']} ({commit['hash']})")
        changelog.append("")

    # Bug Fixes
    if categorized["fixes"]:
        changelog.append("## 🐛 Bug Fixes\n")
        for commit in categorized["fixes"]:
            changelog.append(f"- {commit['subject']} ({commit['hash']})")
        changelog.append("")

    # CI/CD
    if categorized["ci"]:
        changelog.append("## 🔧 CI/CD\n")
        for commit in categorized["ci"]:
            changelog.append(f"- {commit['subject']} ({commit['hash']})")
        changelog.append("")

    # Documentation
    if categorized["docs"]:
        changelog.append("## 📚 Documentation\n")
        for commit in categorized["docs"]:
            changelog.append(f"- {commit['subject']} ({commit['hash']})")
        changelog.append("")

    # Tests
    if categorized["tests"]:
        changelog.append("## 🧪 Tests\n")
        for commit in categorized["tests"]:
            changelog.append(f"- {commit['subject']} ({commit['hash']})")
        changelog.append("")

    # Refactoring
    if categorized["refactor"]:
        changelog.append("## ♻️ Refactoring\n")
        for commit in categorized["refactor"]:
            changelog.append(f"- {commit['subject']} ({commit['hash']})")
        changelog.append("")

    # Other
    if categorized["other"]:
        changelog.append("## 🔀 Other Changes\n")
        for commit in categorized["other"]:
            changelog.append(f"- {commit['subject']} ({commit['hash']})")
        changelog.append("")

    # Contributors
    contributors = set(f"{c['author']}" for c in commits)
    if contributors:
        changelog.append("## 👥 Contributors\n")
        for contributor in sorted(contributors):
            changelog.append(f"- {contributor}")

    return "\n".join(changelog)


def main():
    """Main entry point."""
    version = sys.argv[1] if len(sys.argv) > 1 else None
    changelog = generate_changelog(version)

    # Write to file
    output_file = Path("CHANGELOG.md")
    with open(output_file, "w") as f:
        f.write(changelog)

    print(f"✅ Changelog generated: {output_file}")
    print(f"\n{changelog}")


if __name__ == "__main__":
    main()
