import logging
import subprocess

logger = logging.getLogger(__name__)


def get_git_tag() -> str:
    logger.info("Getting git tag. If doesn't exist, getting commit hash")
    try:
        git_tag = (
            subprocess.check_output(["git", "describe", "--tags", "--always"])
            .decode("utf-8")
            .strip()
        )
        logger.info(f"Identified git tag: {git_tag}")
        return git_tag
    except subprocess.CalledProcessError:
        logger.info("Failed to get git tag, falling back to commit count and hash")
        try:
            count = (
                subprocess.check_output(["git", "rev-list", "--count", "HEAD"])
                .decode("utf-8")
                .strip()
            )
            hash = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
                .decode("utf-8")
                .strip()
            )
            git_tag = f"no-tag-{count}-g{hash}"
            logger.info(f"Created git tag: {git_tag}")
            return git_tag
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create git tag: {e}")
            return "unknown"
