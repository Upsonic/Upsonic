"""Skills container — manages skill loading and provides tools for agents."""

import json
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from .loader import SkillLoader
from .metrics import SkillMetrics
from .skill import Skill
from .utils import is_safe_path, read_file_safe, run_script

if TYPE_CHECKING:
    from .cache import SkillCache

logger = logging.getLogger(__name__)


class Skills:
    """Orchestrates skill loading and provides tools for agents to access skills.

    Responsibilities:

    1. Load skills from one or more :class:`SkillLoader` instances.
    2. Provide accessor methods for loaded skills.
    3. Generate three tool functions that let agents progressively discover
       and use skills.
    4. Produce a system-prompt snippet with available skill metadata.

    Later loaders override earlier ones when skill names conflict.

    Args:
        loaders: List of :class:`SkillLoader` instances to load skills from.

    Example::

        from upsonic import Agent
        from upsonic.skills import Skills, LocalSkills

        agent = Agent(
            model="anthropic/claude-sonnet-4-6",
            skills=Skills(loaders=[
                LocalSkills("/path/to/shared-skills"),
                LocalSkills("/path/to/project-skills"),
            ])
        )
    """

    def __init__(
        self,
        loaders: List[SkillLoader],
        strict_deps: bool = False,
        cache_ttl: Optional[int] = None,
        on_load: Optional[Callable] = None,
        on_script_execute: Optional[Callable] = None,
        on_reference_access: Optional[Callable] = None,
        auto_select: bool = False,
        max_skills: int = 5,
        embedding_provider: Optional[Any] = None,
        policy: Optional[Any] = None,
        knowledge_base: Optional[Any] = None,
    ) -> None:
        self.loaders = loaders
        self.strict_deps = strict_deps
        self._skills: Dict[str, Skill] = {}

        # Caching (Feature 10)
        self._cache: Optional["SkillCache"] = None
        if cache_ttl is not None:
            from .cache import SkillCache

            self._cache = SkillCache(ttl_seconds=cache_ttl)

        # Metrics (Feature 13)
        self._metrics: Dict[str, SkillMetrics] = {}

        # Callbacks (Feature 9)
        self._on_load = on_load
        self._on_script_execute = on_script_execute
        self._on_reference_access = on_reference_access

        # Tool binding (Feature 6)
        self._active_skills: Set[str] = set()

        # Auto-selection (Feature 1)
        self.auto_select = auto_select
        self.max_skills = max_skills
        self.embedding_provider = embedding_provider

        # Safety engine (Feature 8)
        self._policies: List[Any] = []
        if policy is not None:
            if isinstance(policy, list):
                self._policies = policy
            else:
                self._policies = [policy]

        # Knowledge base / RAG integration (Feature 11)
        self._knowledge_base = knowledge_base

        self._load_skills()

    # -- Loading -------------------------------------------------------------

    def _load_skills(self) -> None:
        """Load skills from all loaders, then resolve dependencies."""
        from upsonic.utils.package.exception import SkillValidationError

        for loader in self.loaders:
            try:
                skills = loader.load()
                for skill in skills:
                    if skill.name in self._skills:
                        existing = self._skills[skill.name]
                        logger.warning(
                            "Duplicate skill name '%s' (version %s -> %s), overwriting",
                            skill.name,
                            existing.version or "unknown",
                            skill.version or "unknown",
                        )
                    self._skills[skill.name] = skill
            except SkillValidationError:
                raise
            except Exception as e:
                logger.warning("Error loading skills from %s: %s", loader, e)

        # Dependency resolution (Feature 7)
        if self._skills:
            from .dependency import detect_cycles, get_missing_dependencies

            missing = get_missing_dependencies(self._skills)
            for name, deps in missing.items():
                logger.warning(
                    "Skill '%s' has missing dependencies: %s",
                    name,
                    ", ".join(deps),
                )

            cycles = detect_cycles(self._skills)
            if cycles:
                cycle_strs = [" -> ".join(c) for c in cycles]
                msg = f"Dependency cycles detected: {'; '.join(cycle_strs)}"
                if self.strict_deps:
                    raise SkillValidationError(msg, errors=cycle_strs)
                logger.warning(msg)

        # Initialize metrics for all loaded skills
        for name in self._skills:
            if name not in self._metrics:
                self._metrics[name] = SkillMetrics()

        logger.debug("Loaded %d total skills", len(self._skills))

    def copy(self) -> "Skills":
        """Return a shallow copy that shares loaders and skills but has independent metrics."""
        new = Skills.__new__(Skills)
        new.loaders = self.loaders
        new.strict_deps = self.strict_deps
        new._skills = dict(self._skills)
        new._cache = self._cache
        new._metrics = {name: SkillMetrics() for name in self._skills}
        new._on_load = self._on_load
        new._on_script_execute = self._on_script_execute
        new._on_reference_access = self._on_reference_access
        new._active_skills = set(self._active_skills)
        new.auto_select = self.auto_select
        new.max_skills = self.max_skills
        new.embedding_provider = self.embedding_provider
        new._policies = list(self._policies)
        new._knowledge_base = self._knowledge_base
        return new

    def reload(self) -> None:
        """Reload all skills from loaders (e.g. after filesystem changes)."""
        if self._cache is not None:
            self._cache.invalidate()
        self._skills.clear()
        self._load_skills()

    # -- Accessors -----------------------------------------------------------

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name, or ``None`` if not found."""
        return self._skills.get(name)

    def get_all_skills(self) -> List[Skill]:
        """Return all loaded :class:`Skill` objects."""
        return list(self._skills.values())

    def get_skill_names(self) -> List[str]:
        """Return the names of all loaded skills."""
        return list(self._skills.keys())

    # -- Auto-selection (Feature 1) ------------------------------------------

    def _select_relevant_skills(
        self, task_description: str
    ) -> List[Skill]:
        """Select the most relevant skills for a task using embedding similarity.

        Falls back to returning all skills if no embedding provider is configured.
        """
        if not self.embedding_provider or not self._skills:
            return list(self._skills.values())[:self.max_skills]

        try:
            skill_list = list(self._skills.values())
            texts = [s.description for s in skill_list] + [task_description]

            # Use the embedding provider to get vectors
            embeddings = self.embedding_provider.embed_texts(texts)

            task_vec = embeddings[-1]
            similarities = []
            for i, skill in enumerate(skill_list):
                sim = self._cosine_similarity(embeddings[i], task_vec)
                similarities.append((sim, skill))

            similarities.sort(key=lambda x: x[0], reverse=True)
            return [s for _, s in similarities[:self.max_skills]]
        except Exception as e:
            logger.warning("Auto-selection failed, returning all skills: %s", e)
            return list(self._skills.values())[:self.max_skills]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors (pure Python)."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # -- Safety engine (Feature 8) -------------------------------------------

    def _validate_content(self, content: str, context: str) -> tuple:
        """Validate content against configured policies.

        Returns:
            A tuple of (is_safe: bool, reason: str).
        """
        if not self._policies:
            return True, ""

        try:
            from upsonic.safety_engine.models import PolicyInput

            pi = PolicyInput(input_texts=[content])
            for p in self._policies:
                result = p.check(pi)
                if hasattr(result, 'confidence') and result.confidence > 0.7:
                    reason = getattr(result, 'details', str(result))
                    logger.warning(
                        "Policy blocked skill content (%s): %s", context, reason
                    )
                    return False, reason
        except ImportError:
            logger.debug("Safety engine not available, skipping policy check")
        except Exception as e:
            logger.warning("Policy check error: %s", e)

        return True, ""

    # -- System prompt -------------------------------------------------------

    def get_system_prompt_section(self, task_description: Optional[str] = None) -> str:
        """Generate an XML-formatted system prompt snippet.

        Provides the agent with skill metadata (names, descriptions,
        available scripts/references) so it knows what skills exist
        without loading their full instructions.

        Args:
            task_description: If provided and auto_select is enabled,
                only the most relevant skills are shown.
        """
        if not self._skills:
            return ""

        # Cache key includes task description for auto-select
        cache_key = f"system_prompt:{task_description or ''}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        # Auto-select relevant skills if enabled
        if self.auto_select and task_description:
            selected_skills = self._select_relevant_skills(task_description)
        else:
            selected_skills = list(self._skills.values())

        lines = [
            "<skills_system>",
            "",
            "## What are Skills?",
            "Skills are packages of domain expertise that extend your capabilities. Each skill contains:",
            "- **Instructions**: Detailed guidance on when and how to apply the skill",
            "- **Scripts**: Executable code templates you can use or adapt",
            "- **References**: Supporting documentation (guides, cheatsheets, examples)",
            "- **Assets**: Supporting files like templates, fonts, and icons",
            "",
            "## IMPORTANT: How to Use Skills",
            "**Skill names are NOT callable functions.** You cannot call a skill directly by its name.",
            "Instead, you MUST use the provided skill access tools:",
            "",
            "1. `get_skill_instructions(skill_name)` - Load the full instructions for a skill",
            "2. `get_skill_reference(skill_name, reference_path)` - Access specific documentation",
            "3. `get_skill_script(skill_name, script_path, execute=False)` - Read or run scripts",
            "4. `get_skill_asset(skill_name, asset_path)` - Read asset files (templates, fonts, icons)",
            "",
            "## Progressive Discovery Workflow",
            "1. **Browse**: Review the skill summaries below to understand what's available",
            "2. **Load**: When a task matches a skill, call `get_skill_instructions(skill_name)` first",
            "3. **Reference**: Use `get_skill_reference` to access specific documentation as needed",
            "4. **Scripts**: Use `get_skill_script` to read or execute scripts from a skill",
            "",
            "**IMPORTANT**: References are documentation files (NOT executable). "
            "Only use `get_skill_script` when `<scripts>` lists actual script files. "
            "If `<scripts>none</scripts>`, do NOT call `get_skill_script`.",
            "",
            "This approach ensures you only load detailed instructions when actually needed.",
            "",
            "## Available Skills",
        ]

        for skill in selected_skills:
            lines.append("<skill>")
            lines.append(f"  <name>{skill.name}</name>")
            lines.append(f"  <description>{skill.description}</description>")

            if skill.scripts:
                lines.append(f"  <scripts>{', '.join(skill.scripts)}</scripts>")
            else:
                lines.append("  <scripts>none</scripts>")

            if skill.references:
                lines.append(
                    f"  <references>{', '.join(skill.references)}</references>"
                )

            if skill.assets:
                lines.append(
                    f"  <assets>{', '.join(skill.assets)}</assets>"
                )

            if skill.allowed_tools:
                lines.append(
                    f"  <allowed_tools>{', '.join(skill.allowed_tools)}</allowed_tools>"
                )

            if skill.dependencies:
                lines.append(
                    f"  <dependencies>{', '.join(skill.dependencies)}</dependencies>"
                )

            lines.append("</skill>")

        lines.append("")
        lines.append("</skills_system>")
        result = "\n".join(lines)

        if self._cache is not None:
            self._cache.set(cache_key, result)

        return result

    # -- Tool generation -----------------------------------------------------

    def get_tools(self, prefix: str = "") -> List[Callable[..., str]]:
        """Return three tool functions for agent integration.

        The functions are plain callables that work with Upsonic's
        :class:`~upsonic.tools.ToolProcessor` without any wrappers.

        Args:
            prefix: Optional prefix for tool function names. Use this to
                avoid name collisions when the same tool names are registered
                in different scopes (e.g. ``"task_"`` for task-level tools).
        """

        def get_skill_instructions(skill_name: str) -> str:
            """Load the full instructions for a skill.

            Use this when you need to follow a skill's guidance.

            Args:
                skill_name: The name of the skill to get instructions for.

            Returns:
                A JSON string with the skill's instructions and metadata.
            """
            return self._get_skill_instructions(skill_name)

        def get_skill_reference(
            skill_name: str, reference_path: str
        ) -> str:
            """Load a reference document from a skill's references.

            Use this to access detailed documentation such as style guides,
            API docs, or configuration examples.

            Args:
                skill_name: The name of the skill.
                reference_path: The filename of the reference document.

            Returns:
                A JSON string with the reference content.
            """
            return self._get_skill_reference(skill_name, reference_path)

        def get_skill_script(
            skill_name: str,
            script_path: str,
            execute: bool = False,
            args: Optional[List[str]] = None,
            timeout: int = 30,
        ) -> str:
            """Read or execute a script from a skill.

            Set ``execute=True`` to run the script and get its output,
            or ``execute=False`` (default) to read the script content.

            Args:
                skill_name: The name of the skill.
                script_path: The filename of the script.
                execute: If True, execute the script. If False, return content.
                args: Optional arguments to pass to the script (only when execute=True).
                timeout: Maximum execution time in seconds (default: 30).

            Returns:
                A JSON string with either script content or execution results.
            """
            return self._get_skill_script(
                skill_name, script_path, execute, args, timeout
            )

        def get_skill_asset(
            skill_name: str, asset_path: str
        ) -> str:
            """Read an asset file from a skill's assets directory.

            Assets include templates, fonts, icons, and other supporting files
            that complement the skill's instructions and scripts.

            Args:
                skill_name: The name of the skill.
                asset_path: The filename of the asset.

            Returns:
                A JSON string with the asset content or metadata.
            """
            return self._get_skill_asset(skill_name, asset_path)

        tools = [get_skill_instructions, get_skill_reference, get_skill_script, get_skill_asset]

        # Knowledge base search tool (Feature 11)
        if self._knowledge_base is not None:
            kb = self._knowledge_base

            def search_skill_references(query: str, top_k: int = 5) -> str:
                """Semantically search across all skill reference documents.

                Use this when you need to find specific information across
                multiple skills' references without knowing which skill has it.

                Args:
                    query: The search query string.
                    top_k: Maximum number of results to return (default: 5).

                Returns:
                    A JSON string with search results.
                """
                try:
                    results = kb.query(query, top_k=top_k)
                    return json.dumps(
                        {
                            "query": query,
                            "results": [
                                r.to_dict() if hasattr(r, 'to_dict') else str(r)
                                for r in results
                            ],
                        }
                    )
                except Exception as e:
                    return json.dumps({"error": f"Search failed: {e}"})

            tools.append(search_skill_references)

        # Apply prefix to tool function names if provided
        if prefix:
            for tool in tools:
                tool.__name__ = f"{prefix}{tool.__name__}"
                tool.__qualname__ = f"{prefix}{tool.__qualname__}"

        return tools

    # -- Private tool implementations ----------------------------------------

    def _get_skill_instructions(self, skill_name: str) -> str:
        # Check cache first
        if self._cache is not None:
            cache_key = f"instructions:{skill_name}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        skill = self.get_skill(skill_name)
        if skill is None:
            return json.dumps(
                {
                    "error": f"Skill '{skill_name}' not found",
                    "available_skills": ", ".join(self.get_skill_names()),
                }
            )

        # Tool binding: track active skills
        self._active_skills.add(skill_name)

        result_dict = {
            "skill_name": skill.name,
            "description": skill.description,
            "instructions": skill.instructions,
            "available_scripts": skill.scripts,
            "available_references": skill.references,
            "available_assets": skill.assets,
            "dependencies": skill.dependencies,
            "version": skill.version,
        }
        if skill.allowed_tools:
            result_dict["recommended_tools"] = skill.allowed_tools

        result = json.dumps(result_dict)

        # Safety check
        is_safe, reason = self._validate_content(
            skill.instructions, f"skill:{skill_name}:instructions"
        )
        if not is_safe:
            return json.dumps(
                {"error": f"Content blocked by policy: {reason}", "skill_name": skill_name}
            )

        # Metrics
        if skill_name in self._metrics:
            self._metrics[skill_name].record_load(chars=len(result))

        # Callbacks
        self._invoke_load_callback(skill)

        if self._cache is not None:
            self._cache.set(f"instructions:{skill_name}", result)

        return result

    def _get_skill_reference(
        self, skill_name: str, reference_path: str
    ) -> str:
        skill = self.get_skill(skill_name)
        if skill is None:
            return json.dumps(
                {
                    "error": f"Skill '{skill_name}' not found",
                    "available_skills": ", ".join(self.get_skill_names()),
                }
            )

        if reference_path not in skill.references:
            return json.dumps(
                {
                    "error": f"Reference '{reference_path}' not found in skill '{skill_name}'",
                    "available_references": skill.references,
                }
            )

        refs_dir = Path(skill.source_path) / "references"
        if not is_safe_path(refs_dir, reference_path):
            return json.dumps(
                {
                    "error": f"Invalid reference path: '{reference_path}'",
                    "skill_name": skill_name,
                }
            )

        try:
            content = read_file_safe(refs_dir / reference_path)

            # Safety check
            is_safe, reason = self._validate_content(
                content, f"skill:{skill_name}:reference:{reference_path}"
            )
            if not is_safe:
                return json.dumps(
                    {"error": f"Content blocked by policy: {reason}", "skill_name": skill_name}
                )

            result = json.dumps(
                {
                    "skill_name": skill_name,
                    "reference_path": reference_path,
                    "content": content,
                }
            )

            # Metrics
            if skill_name in self._metrics:
                self._metrics[skill_name].record_reference_access(chars=len(content))

            # Callbacks
            self._invoke_reference_callback(skill_name, reference_path)

            return result
        except Exception as e:
            return json.dumps(
                {
                    "error": f"Error reading reference file: {e}",
                    "skill_name": skill_name,
                    "reference_path": reference_path,
                }
            )

    def _get_skill_script(
        self,
        skill_name: str,
        script_path: str,
        execute: bool = False,
        args: Optional[List[str]] = None,
        timeout: int = 30,
    ) -> str:
        skill = self.get_skill(skill_name)
        if skill is None:
            return json.dumps(
                {
                    "error": f"Skill '{skill_name}' not found",
                    "available_skills": ", ".join(self.get_skill_names()),
                }
            )

        if script_path not in skill.scripts:
            return json.dumps(
                {
                    "error": f"Script '{script_path}' not found in skill '{skill_name}'",
                    "available_scripts": skill.scripts,
                }
            )

        scripts_dir = Path(skill.source_path) / "scripts"
        if not is_safe_path(scripts_dir, script_path):
            return json.dumps(
                {
                    "error": f"Invalid script path: '{script_path}'",
                    "skill_name": skill_name,
                }
            )

        script_file = scripts_dir / script_path

        if not execute:
            try:
                content = read_file_safe(script_file)
                return json.dumps(
                    {
                        "skill_name": skill_name,
                        "script_path": script_path,
                        "content": content,
                    }
                )
            except Exception as e:
                return json.dumps(
                    {
                        "error": f"Error reading script file: {e}",
                        "skill_name": skill_name,
                        "script_path": script_path,
                    }
                )

        # Execute mode
        try:
            result = run_script(
                script_path=script_file,
                args=args,
                timeout=timeout,
                cwd=Path(skill.source_path),
            )

            # Metrics
            if skill_name in self._metrics:
                self._metrics[skill_name].record_script_execution()

            # Callbacks
            self._invoke_script_callback(skill_name, script_path, result.returncode)

            return json.dumps(
                {
                    "skill_name": skill_name,
                    "script_path": script_path,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }
            )
        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "error": f"Script execution timed out after {timeout} seconds",
                    "skill_name": skill_name,
                    "script_path": script_path,
                }
            )
        except FileNotFoundError as e:
            return json.dumps(
                {
                    "error": f"Interpreter or script not found: {e}",
                    "skill_name": skill_name,
                    "script_path": script_path,
                }
            )
        except Exception as e:
            return json.dumps(
                {
                    "error": f"Error executing script: {e}",
                    "skill_name": skill_name,
                    "script_path": script_path,
                }
            )

    def _get_skill_asset(self, skill_name: str, asset_path: str) -> str:
        skill = self.get_skill(skill_name)
        if skill is None:
            return json.dumps(
                {
                    "error": f"Skill '{skill_name}' not found",
                    "available_skills": ", ".join(self.get_skill_names()),
                }
            )

        if asset_path not in skill.assets:
            return json.dumps(
                {
                    "error": f"Asset '{asset_path}' not found in skill '{skill_name}'",
                    "available_assets": skill.assets,
                }
            )

        assets_dir = Path(skill.source_path) / "assets"
        if not is_safe_path(assets_dir, asset_path):
            return json.dumps(
                {
                    "error": f"Invalid asset path: '{asset_path}'",
                    "skill_name": skill_name,
                }
            )

        try:
            content = read_file_safe(assets_dir / asset_path)
            return json.dumps(
                {
                    "skill_name": skill_name,
                    "asset_path": asset_path,
                    "content": content,
                }
            )
        except Exception as e:
            return json.dumps(
                {
                    "error": f"Error reading asset file: {e}",
                    "skill_name": skill_name,
                    "asset_path": asset_path,
                }
            )

    # -- Metrics (Feature 13) ------------------------------------------------

    def get_metrics(self) -> Dict[str, SkillMetrics]:
        """Return usage metrics for all skills."""
        return dict(self._metrics)

    # -- Tool binding (Feature 6) --------------------------------------------

    def get_active_skill_tools(self) -> Set[str]:
        """Return the union of ``allowed_tools`` from all actively used skills."""
        tools: Set[str] = set()
        for name in self._active_skills:
            skill = self._skills.get(name)
            if skill and skill.allowed_tools:
                tools.update(skill.allowed_tools)
        return tools

    # -- Callbacks (Feature 9) ------------------------------------------------

    def _invoke_load_callback(self, skill: Skill) -> None:
        if self._on_load is not None:
            try:
                self._on_load(skill.name, skill.description)
            except Exception as e:
                logger.warning("on_load callback error: %s", e)

    def _invoke_reference_callback(
        self, skill_name: str, reference_path: str
    ) -> None:
        if self._on_reference_access is not None:
            try:
                self._on_reference_access(skill_name, reference_path)
            except Exception as e:
                logger.warning("on_reference_access callback error: %s", e)

    def _invoke_script_callback(
        self, skill_name: str, script_path: str, returncode: Optional[int]
    ) -> None:
        if self._on_script_execute is not None:
            try:
                self._on_script_execute(skill_name, script_path, returncode)
            except Exception as e:
                logger.warning("on_script_execute callback error: %s", e)

    # -- Merge (Feature 5) ---------------------------------------------------

    @classmethod
    def merge(cls, *instances: "Skills") -> "Skills":
        """Create a new Skills instance by merging multiple instances.

        Later instances override earlier ones when skill names conflict.
        The merged instance has no loaders — it's a snapshot.
        """
        from .loader import InlineSkills

        combined: Dict[str, Skill] = {}
        for inst in instances:
            for name, skill in inst._skills.items():
                combined[name] = skill

        return cls(loaders=[InlineSkills(list(combined.values()))])

    # -- Dunder methods ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, skill_name: str) -> bool:
        return skill_name in self._skills

    def __repr__(self) -> str:
        names = ", ".join(self.get_skill_names())
        return f"Skills([{names}])"
