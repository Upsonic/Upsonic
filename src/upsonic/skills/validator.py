"""Skill validation logic following the Agent Skills spec."""

import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

MAX_SKILL_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
MAX_COMPATIBILITY_LENGTH = 500

ALLOWED_FIELDS = {
    "name",
    "description",
    "version",
    "license",
    "allowed-tools",
    "metadata",
    "compatibility",
    "dependencies",
}



def _validate_name(name: str, skill_dir: Optional[Path] = None) -> List[str]:
    errors: List[str] = []

    if not name or not isinstance(name, str) or not name.strip():
        errors.append("Field 'name' must be a non-empty string")
        return errors

    name = unicodedata.normalize("NFKC", name.strip())

    if len(name) > MAX_SKILL_NAME_LENGTH:
        errors.append(
            f"Skill name '{name}' exceeds {MAX_SKILL_NAME_LENGTH} character limit "
            f"({len(name)} chars)"
        )

    if name != name.lower():
        errors.append(f"Skill name '{name}' must be lowercase")

    if name.startswith("-") or name.endswith("-"):
        errors.append("Skill name cannot start or end with a hyphen")

    if "--" in name:
        errors.append("Skill name cannot contain consecutive hyphens")

    if not all(c.isalnum() or c == "-" for c in name):
        errors.append(
            f"Skill name '{name}' contains invalid characters. "
            "Only letters, digits, and hyphens are allowed."
        )

    if skill_dir:
        dir_name = unicodedata.normalize("NFKC", skill_dir.name)
        if dir_name != name:
            errors.append(
                f"Directory name '{dir_name}' must match skill name '{name}'"
            )

    return errors


def _validate_description(description: str) -> List[str]:
    errors: List[str] = []

    if not description or not isinstance(description, str) or not description.strip():
        errors.append("Field 'description' must be a non-empty string")
        return errors

    if len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(
            f"Description exceeds {MAX_DESCRIPTION_LENGTH} character limit "
            f"({len(description)} chars)"
        )

    # Security: no XML tags in description (prevents prompt injection)
    if "<" in description or ">" in description:
        errors.append(
            "Description must not contain XML bracket characters (< or >). "
            "Use plain text only."
        )

    return errors


def _validate_compatibility(compatibility: str) -> List[str]:
    errors: List[str] = []
    if not isinstance(compatibility, str):
        errors.append("Field 'compatibility' must be a string")
        return errors
    if len(compatibility) > MAX_COMPATIBILITY_LENGTH:
        errors.append(
            f"Compatibility exceeds {MAX_COMPATIBILITY_LENGTH} character limit "
            f"({len(compatibility)} chars)"
        )
    return errors


def _validate_license(license_val: str) -> List[str]:
    if not isinstance(license_val, str):
        return ["Field 'license' must be a string"]
    return []


def _validate_allowed_tools(allowed_tools: object) -> List[str]:
    if not isinstance(allowed_tools, list):
        return ["Field 'allowed-tools' must be a list"]
    if not all(isinstance(t, str) for t in allowed_tools):
        return ["Field 'allowed-tools' must be a list of strings"]
    return []


def _validate_dependencies(dependencies: object) -> List[str]:
    if not isinstance(dependencies, list):
        return ["Field 'dependencies' must be a list"]
    if not all(isinstance(d, str) for d in dependencies):
        return ["Field 'dependencies' must be a list of strings"]
    return []


def _validate_metadata_value(metadata_val: object) -> List[str]:
    if not isinstance(metadata_val, dict):
        return ["Field 'metadata' must be a dictionary"]
    return []


def _validate_metadata_fields(metadata: Dict) -> List[str]:
    extra = set(metadata.keys()) - ALLOWED_FIELDS
    if extra:
        return [
            f"Unexpected fields in frontmatter: {', '.join(sorted(extra))}. "
            f"Only {sorted(ALLOWED_FIELDS)} are allowed."
        ]
    return []



def validate_metadata(
    metadata: Dict,
    skill_dir: Optional[Path] = None,
) -> List[str]:
    """Validate parsed skill frontmatter metadata.

    Args:
        metadata: Parsed YAML frontmatter dictionary.
        skill_dir: Optional skill directory path (for name-directory match).

    Returns:
        List of validation error messages. Empty means valid.
    """
    errors: List[str] = []
    errors.extend(_validate_metadata_fields(metadata))

    if "name" not in metadata:
        errors.append("Missing required field in frontmatter: name")
    else:
        errors.extend(_validate_name(metadata["name"], skill_dir))

    if "description" not in metadata:
        errors.append("Missing required field in frontmatter: description")
    else:
        errors.extend(_validate_description(metadata["description"]))

    if "compatibility" in metadata:
        errors.extend(_validate_compatibility(metadata["compatibility"]))
    if "license" in metadata:
        errors.extend(_validate_license(metadata["license"]))
    if "allowed-tools" in metadata:
        errors.extend(_validate_allowed_tools(metadata["allowed-tools"]))
    if "metadata" in metadata:
        errors.extend(_validate_metadata_value(metadata["metadata"]))
    if "dependencies" in metadata:
        errors.extend(_validate_dependencies(metadata["dependencies"]))

    return errors


def validate_skill_directory(skill_dir: Path) -> List[str]:
    """Validate a skill directory structure and contents.

    Parses the SKILL.md frontmatter and runs all validation checks.

    Args:
        skill_dir: Path to the skill directory.

    Returns:
        List of validation error messages. Empty means valid.
    """
    from upsonic.utils.package.exception import SkillParseError

    skill_dir = Path(skill_dir)

    if not skill_dir.exists():
        return [f"Path does not exist: {skill_dir}"]
    if not skill_dir.is_dir():
        return [f"Not a directory: {skill_dir}"]

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return ["Missing required file: SKILL.md"]

    try:
        content = skill_md.read_text(encoding="utf-8")

        if not content.startswith("---"):
            raise SkillParseError("SKILL.md must start with YAML frontmatter (---)")

        parts = content.split("---", 2)
        if len(parts) < 3:
            raise SkillParseError(
                "SKILL.md frontmatter not properly closed with ---"
            )

        frontmatter_str = parts[1]

        # Try yaml.safe_load, fall back to simple parser
        try:
            import yaml

            metadata = yaml.safe_load(frontmatter_str)
        except ImportError:
            metadata = _simple_yaml_parse(frontmatter_str)
        except Exception as e:
            return [f"Invalid YAML in frontmatter: {e}"]

        if not isinstance(metadata, dict):
            raise SkillParseError("SKILL.md frontmatter must be a YAML mapping")

    except SkillParseError as e:
        return [str(e)]
    except Exception as e:
        return [f"Error reading SKILL.md: {e}"]

    return validate_metadata(metadata, skill_dir)


def _simple_yaml_parse(text: str) -> Dict:
    """Minimal fallback parser for key: value frontmatter."""
    result: Dict = {}
    for line in text.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value.startswith("[") and value.endswith("]"):
                result[key] = [
                    v.strip().strip('"').strip("'")
                    for v in value[1:-1].split(",")
                    if v.strip()
                ]
            elif value:
                result[key] = value
    return result
