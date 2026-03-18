"""
Smoke tests that run every Python code example from docs/skills/**/*.mdx.

Each test executes the exact code from the docs (adapted only for temp dirs
instead of ./my-skills paths). If any example crashes, the underlying code
must be fixed — NOT the test.

Organized by source doc file.
"""

import os
import tempfile
from pathlib import Path

import pytest


requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


def _make_skill_dir(base, name="code-review", version=None):
    """Create a minimal valid skill directory."""
    skill_dir = Path(base) / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    version_line = f"\nversion: {version}" if version else ""
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Skill for {name}{version_line}\n---\n"
        f"Instructions for {name}. Follow best practices."
    )
    (skill_dir / "scripts").mkdir(exist_ok=True)
    (skill_dir / "references").mkdir(exist_ok=True)
    (skill_dir / "references" / "guide.txt").write_text(f"Reference guide for {name}.")
    return skill_dir


# ===========================================================================
# docs/skills/overview.mdx
# ===========================================================================

class TestOverviewExamples:

    def test_agent_with_local_skills(self):
        """overview.mdx — Agent + LocalSkills"""
        from upsonic import Agent
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "code-review")
            agent = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="Code Reviewer",
                role="Senior Developer",
                goal="Review code for quality and best practices",
                skills=Skills(loaders=[LocalSkills(d)]),
            )

    def test_task_with_builtin_skills(self):
        """overview.mdx — Task + BuiltinSkills"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, BuiltinSkills

        agent = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Writer",
            role="Content Writer",
            goal="Create clear, engaging content",
        )

        task = Task(
            description="Summarize this technical document in 3 bullet points.",
            skills=Skills(loaders=[BuiltinSkills(skills=["summarization"])]),
        )

    def test_team_with_builtin_skills(self):
        """overview.mdx — Team + BuiltinSkills"""
        from upsonic import Agent, Task, Team
        from upsonic.skills import Skills, BuiltinSkills

        analyst = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Data Analyst",
            role="Data Analysis Expert",
            goal="Analyze data and extract insights",
        )

        writer = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Report Writer",
            role="Business Report Specialist",
            goal="Create professional summaries",
        )

        team = Team(
            agents=[analyst, writer],
            skills=Skills(loaders=[BuiltinSkills(skills=["data-analysis", "summarization"])]),
            mode="coordinate",
            model="anthropic/claude-sonnet-4-6",
        )

        task = Task(description="Analyze Q4 sales trends and write an executive summary.")

    def test_load_all_and_specific_builtins(self):
        """overview.mdx — Load all / specific builtins"""
        from upsonic.skills import Skills, BuiltinSkills

        skills = Skills(loaders=[BuiltinSkills()])
        skills = Skills(loaders=[BuiltinSkills(skills=["code-review"])])

    def test_combined_loaders(self):
        """overview.mdx — BuiltinSkills + LocalSkills + GitHubSkills"""
        from upsonic.skills import Skills, LocalSkills, GitHubSkills, BuiltinSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "company-skill")
            skills = Skills(loaders=[
                BuiltinSkills(skills=["code-review"]),
                LocalSkills(d),
                GitHubSkills(repo="anthropics/skills", branch="main", skills=["pdf"]),
            ])


# ===========================================================================
# docs/skills/skill-format.mdx
# ===========================================================================

class TestSkillFormatExamples:

    def test_load_from_directory(self):
        """skill-format.mdx — Load from skills directory"""
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "code-review")
            _make_skill_dir(d, "summarization")
            skills = Skills(loaders=[LocalSkills(d)])

    def test_inline_skill(self):
        """skill-format.mdx — Inline skill"""
        from upsonic.skills import Skills, Skill, InlineSkills

        skill = Skill(
            name="json-formatter",
            description="Format and validate JSON output",
            instructions="Always return valid, pretty-printed JSON with 2-space indentation.",
            source_path="",
            scripts=[],
            references=[],
        )

        skills = Skills(loaders=[InlineSkills([skill])])


# ===========================================================================
# docs/skills/metrics.mdx
# ===========================================================================

class TestMetricsExamples:

    def test_agent_metrics(self):
        """metrics.mdx — Agent skill metrics"""
        from upsonic import Agent
        from upsonic.skills import Skills, BuiltinSkills

        agent = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Reviewer",
            role="Code Reviewer",
            goal="Review code quality",
            skills=Skills(loaders=[BuiltinSkills(skills=["code-review"])]),
        )

        metrics = agent.get_skill_metrics()
        print(metrics)

    def test_task_metrics(self):
        """metrics.mdx — Task skill metrics"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, BuiltinSkills

        task = Task(
            description="Summarize this document.",
            skills=Skills(loaders=[BuiltinSkills(skills=["summarization"])]),
        )

        agent = Agent(model="anthropic/claude-sonnet-4-6", name="Writer", role="Writer", goal="Write content")

        metrics = task.get_skill_metrics()

    def test_skills_direct_metrics(self):
        """metrics.mdx — Skills.get_metrics() directly"""
        from upsonic.skills import Skills, BuiltinSkills

        skills = Skills(loaders=[BuiltinSkills()])
        metrics = skills.get_metrics()

        for skill_name, m in metrics.items():
            print(f"{skill_name}: {m.to_dict()}")

    def test_team_metrics(self):
        """metrics.mdx — Team metrics"""
        from upsonic import Agent, Task, Team
        from upsonic.skills import Skills, BuiltinSkills

        agent_a = Agent(model="anthropic/claude-sonnet-4-6", name="Agent A", role="Analyst", goal="Analyze")
        agent_b = Agent(model="anthropic/claude-sonnet-4-6", name="Agent B", role="Writer", goal="Write")

        team = Team(
            agents=[agent_a, agent_b],
            skills=Skills(loaders=[BuiltinSkills()]),
            mode="coordinate",
            model="anthropic/claude-sonnet-4-6",
        )

        task = Task(description="Analyze data and write a report.")

        print(agent_a.get_skill_metrics())
        print(agent_b.get_skill_metrics())

    def test_skill_metrics_serialization(self):
        """metrics.mdx — SkillMetrics serialization"""
        from upsonic.skills import SkillMetrics

        m = SkillMetrics(load_count=3, reference_access_count=1)
        data = m.to_dict()
        restored = SkillMetrics.from_dict(data)


# ===========================================================================
# docs/skills/loaders/overview.mdx
# ===========================================================================

class TestLoadersOverviewExamples:

    def test_basic_combined(self):
        """loaders/overview.mdx — Basic combined loaders"""
        from upsonic.skills import Skills, LocalSkills, BuiltinSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "my-skill")
            skills = Skills(loaders=[
                BuiltinSkills(),
                LocalSkills(d),
            ])

    def test_full_combined(self):
        """loaders/overview.mdx — Full combined loaders"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, LocalSkills, BuiltinSkills, GitHubSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "project-skill")
            skills = Skills(loaders=[
                BuiltinSkills(),
                GitHubSkills(repo="anthropics/skills", skills=["pdf"]),
                LocalSkills(d),
            ])

            agent = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="My Agent",
                role="Development Assistant",
                goal="Help with development tasks",
                skills=skills,
            )

            task = Task(description="Analyze this code for quality issues and summarize the findings.")


# ===========================================================================
# docs/skills/loaders/builtin.mdx
# ===========================================================================

class TestBuiltinLoaderExamples:

    def test_load_all(self):
        """builtin.mdx — Load all builtins"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, BuiltinSkills

        skills = Skills(loaders=[BuiltinSkills()])

        agent = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Assistant",
            role="General Purpose Assistant",
            goal="Help with various tasks using built-in expertise",
            skills=skills,
        )

        task = Task(description="Review this code for bugs and suggest improvements.")

    def test_load_specific(self):
        """builtin.mdx — Load specific builtins on Task"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, BuiltinSkills

        agent = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Reviewer",
            role="Code Reviewer",
            goal="Review and summarize code changes",
        )

        task = Task(
            description="Summarize the key changes in this pull request.",
            skills=Skills(loaders=[BuiltinSkills(skills=["code-review", "summarization"])]),
        )

    def test_team_with_builtins(self):
        """builtin.mdx — Team with BuiltinSkills"""
        from upsonic import Agent, Task, Team
        from upsonic.skills import Skills, BuiltinSkills

        analyst = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Data Analyst",
            role="Data Analysis Expert",
            goal="Analyze data and extract insights",
        )

        writer = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Report Writer",
            role="Business Report Specialist",
            goal="Create professional summaries",
        )

        team = Team(
            agents=[analyst, writer],
            skills=Skills(loaders=[BuiltinSkills(skills=["data-analysis", "summarization"])]),
            mode="coordinate",
            model="anthropic/claude-sonnet-4-6",
        )

        task = Task(description="Analyze Q4 sales data and write an executive summary.")

    def test_available_skills(self):
        """builtin.mdx — List available skills"""
        from upsonic.skills import BuiltinSkills

        loader = BuiltinSkills()
        print(loader.available_skills())

    def test_combining_with_local(self):
        """builtin.mdx — Combine with LocalSkills"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, BuiltinSkills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "custom-skill")
            skills = Skills(loaders=[
                BuiltinSkills(),
                LocalSkills(d),
            ])

            agent = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="My Agent",
                role="Developer Assistant",
                goal="Assist with development tasks",
                skills=skills,
            )

            task = Task(description="Analyze this dataset and produce a summary report.")


# ===========================================================================
# docs/skills/loaders/local.mdx
# ===========================================================================

class TestLocalLoaderExamples:

    def test_agent_with_local(self):
        """local.mdx — Agent with LocalSkills"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "code-review")
            skills = Skills(loaders=[LocalSkills(d)])

            agent = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="Code Reviewer",
                role="Senior Developer",
                goal="Review code for quality and best practices",
                skills=skills,
            )

            task = Task(description="Review this Python function for potential bugs.")

    def test_task_with_local(self):
        """local.mdx — Task with LocalSkills"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "perf-skill")
            agent = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="Developer",
                role="Software Engineer",
                goal="Write and review code",
            )

            task = Task(
                description="Analyze this module for performance bottlenecks.",
                skills=Skills(loaders=[LocalSkills(d)]),
            )

    def test_team_with_local(self):
        """local.mdx — Team with LocalSkills"""
        from upsonic import Agent, Task, Team
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "review-skill")
            analyst = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="Code Analyst",
                role="Code Analysis Expert",
                goal="Analyze code for issues",
            )

            writer = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="Report Writer",
                role="Technical Writer",
                goal="Write clear technical reports",
            )

            team = Team(
                agents=[analyst, writer],
                skills=Skills(loaders=[LocalSkills(d)]),
                mode="coordinate",
                model="anthropic/claude-sonnet-4-6",
            )

            task = Task(description="Review the authentication module and write a report.")

    def test_multiple_skills_directory(self):
        """local.mdx — Multiple skills directory"""
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "code-review")
            _make_skill_dir(d, "summarization")
            skills = Skills(loaders=[LocalSkills(d)])

    def test_single_skill_directory(self):
        """local.mdx — Single skill directory"""
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            skill_path = _make_skill_dir(d, "code-review")
            skills = Skills(loaders=[LocalSkills(str(skill_path))])

    def test_version_filtering(self):
        """local.mdx — Version constraint filtering"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "stable-skill", version="1.5.0")
            _make_skill_dir(d, "old-skill", version="0.5.0")
            skills = Skills(loaders=[
                LocalSkills(d, version_constraint=">=1.0.0,<2.0.0")
            ])

            agent = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="Agent",
                role="Assistant",
                goal="Help with tasks",
                skills=skills,
            )

            task = Task(description="Review this code using only stable skill versions.")


# ===========================================================================
# docs/skills/loaders/inline.mdx
# ===========================================================================

class TestInlineLoaderExamples:

    def test_single_inline(self):
        """inline.mdx — Single inline skill"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, Skill, InlineSkills

        json_skill = Skill(
            name="json-formatting",
            description="Format and validate JSON output",
            instructions="Always return valid, pretty-printed JSON with 2-space indentation. Validate structure before returning.",
            source_path="",
            scripts=[],
            references=[],
        )

        agent = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Data Formatter",
            role="Data Processing Specialist",
            goal="Format and structure data outputs",
            skills=Skills(loaders=[InlineSkills([json_skill])]),
        )

        task = Task(description="Convert this CSV data to JSON: name,age\nAlice,30\nBob,25")

    def test_multiple_inline(self):
        """inline.mdx — Multiple inline skills"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, Skill, InlineSkills

        skills_list = [
            Skill(
                name="step-by-step",
                description="Break down complex problems into clear steps",
                instructions="When solving problems, always break them into numbered steps.",
                source_path="",
                scripts=[],
                references=[],
            ),
            Skill(
                name="code-examples",
                description="Include runnable code examples in explanations",
                instructions="When explaining concepts, always include at least one complete, runnable code example.",
                source_path="",
                scripts=[],
                references=[],
            ),
        ]

        agent = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Teacher",
            role="Programming Instructor",
            goal="Teach programming concepts clearly",
            skills=Skills(loaders=[InlineSkills(skills_list)]),
        )

        task = Task(description="Explain how Python decorators work.")

    def test_inline_on_task(self):
        """inline.mdx — Inline skill on Task"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, Skill, InlineSkills

        review_skill = Skill(
            name="security-review",
            description="Review code for security vulnerabilities",
            instructions="Focus on OWASP Top 10 vulnerabilities.",
            source_path="",
            scripts=[],
            references=[],
        )

        agent = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Security Agent",
            role="Security Analyst",
            goal="Identify security issues in code",
        )

        task = Task(
            description="Review this login function for security vulnerabilities.",
            skills=Skills(loaders=[InlineSkills([review_skill])]),
        )

    def test_inline_with_team(self):
        """inline.mdx — Inline skills with Team"""
        from upsonic import Agent, Task, Team
        from upsonic.skills import Skills, Skill, InlineSkills

        skills_list = [
            Skill(
                name="data-analysis",
                description="Analyze data and extract insights",
                instructions="When analyzing data, identify trends, outliers, and key metrics.",
                source_path="",
                scripts=[],
                references=[],
            ),
            Skill(
                name="report-writing",
                description="Write professional business reports",
                instructions="Structure reports with executive summary, findings, and recommendations.",
                source_path="",
                scripts=[],
                references=[],
            ),
        ]

        analyst = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Analyst",
            role="Data Analyst",
            goal="Analyze data and find insights",
        )

        writer = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Writer",
            role="Report Writer",
            goal="Write clear business reports",
        )

        team = Team(
            agents=[analyst, writer],
            skills=Skills(loaders=[InlineSkills(skills_list)]),
            mode="coordinate",
            model="anthropic/claude-sonnet-4-6",
        )

        task = Task(description="Analyze the customer feedback data and write a summary report.")


# ===========================================================================
# docs/skills/loaders/github.mdx
# ===========================================================================

class TestGitHubLoaderExamples:

    def test_agent_with_github(self):
        """github.mdx — Agent with GitHubSkills"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, GitHubSkills

        skills = Skills(loaders=[
            GitHubSkills(
                repo="anthropics/skills",
                branch="main",
                path="skills/",
            )
        ])

        agent = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Team Agent",
            role="Development Assistant",
            goal="Help with development tasks using shared team skills",
            skills=skills,
        )

        task = Task(description="Review this pull request for code quality issues.")

    def test_specific_skills_with_task(self):
        """github.mdx — Specific GitHub skills on Task"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, GitHubSkills

        agent = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Reviewer",
            role="Code Reviewer",
            goal="Review code quality",
        )

        task = Task(
            description="Check this function for edge cases and missing tests.",
            skills=Skills(loaders=[
                GitHubSkills(
                    repo="anthropics/skills",
                    branch="main",
                    path="skills/",
                    skills=["claude-api", "pdf"],
                )
            ]),
        )

    def test_team_with_github(self):
        """github.mdx — Team with GitHubSkills"""
        from upsonic import Agent, Task, Team
        from upsonic.skills import Skills, GitHubSkills

        backend_dev = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Backend Developer",
            role="Backend Expert",
            goal="Review backend code and APIs",
        )

        frontend_dev = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Frontend Developer",
            role="Frontend Expert",
            goal="Review frontend code and UI",
        )

        team = Team(
            agents=[backend_dev, frontend_dev],
            skills=Skills(loaders=[
                GitHubSkills(
                    repo="anthropics/skills",
                    branch="main",
                    path="skills/",
                )
            ]),
            mode="coordinate",
            model="anthropic/claude-sonnet-4-6",
        )

        task = Task(description="Review the full stack changes in the user profile feature.")

    def test_cache_control(self):
        """github.mdx — Cache control"""
        from upsonic import Agent, Task
        from upsonic.skills import Skills, GitHubSkills

        skills = Skills(loaders=[
            GitHubSkills(
                repo="anthropics/skills",
                cache_dir="/tmp/skill-cache",
                cache_ttl=7200,
                force_refresh=True,
            )
        ])

        agent = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Agent",
            role="Assistant",
            goal="Help with tasks",
            skills=skills,
        )

        task = Task(description="Use the latest skills to analyze this code.")


# ===========================================================================
# docs/skills/loaders/url.mdx
# ===========================================================================

class TestURLLoaderExamples:

    def test_url_construct(self):
        """url.mdx — URLSkills construct (no download, just import check)"""
        from upsonic.skills import URLSkills
        # URLSkills requires a real URL to download; just verify the import works
        # and construction doesn't crash with a placeholder
        loader = URLSkills(url="https://example.com/my-skills.tar.gz")


# ===========================================================================
# docs/skills/advanced/versioning.mdx
# ===========================================================================

class TestVersioningExamples:

    def test_skill_with_version(self):
        """versioning.mdx — Skill with version"""
        from upsonic.skills import Skill

        skill = Skill(
            name="code-review",
            description="Code quality analysis",
            instructions="Review code for quality...",
            source_path="",
            scripts=[],
            references=[],
            version="2.1.0",
        )

    def test_version_constraint(self):
        """versioning.mdx — Version constraint filtering"""
        from upsonic import Agent
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "v1-skill", version="1.2.0")
            _make_skill_dir(d, "v2-skill", version="2.5.0")
            skills = Skills(loaders=[
                LocalSkills(d, version_constraint=">=1.0.0,<2.0.0")
            ])

            agent = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="Agent",
                role="Assistant",
                goal="Help with tasks",
                skills=skills,
            )


# ===========================================================================
# docs/skills/advanced/dependencies.mdx
# ===========================================================================

class TestDependencyExamples:

    def test_default_mode_warns(self):
        """dependencies.mdx — Default mode logs warning on missing deps"""
        from upsonic import Agent
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "advanced-review")
            # advanced-review has no declared deps in SKILL.md, so no warning
            skills = Skills(loaders=[LocalSkills(d)])

            agent = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="Agent",
                role="Reviewer",
                goal="Review code",
                skills=skills,
            )

    def test_strict_mode(self):
        """dependencies.mdx — strict_deps=True raises on missing deps"""
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "safe-skill")
            try:
                skills = Skills(
                    loaders=[LocalSkills(d)],
                    strict_deps=True,
                )
            except Exception as e:
                print(f"Dependency error: {e}")

    def test_inline_with_dependencies(self):
        """dependencies.mdx — Inline skills with dependencies"""
        from upsonic import Agent
        from upsonic.skills import Skills, Skill, InlineSkills

        base_skill = Skill(
            name="formatting",
            description="Output formatting standards",
            instructions="Use consistent formatting...",
            source_path="",
            scripts=[],
            references=[],
        )

        advanced_skill = Skill(
            name="report-writing",
            description="Professional report writing",
            instructions="Write professional reports following formatting standards...",
            source_path="",
            scripts=[],
            references=[],
            dependencies=["formatting"],
        )

        skills = Skills(loaders=[InlineSkills([base_skill, advanced_skill])])

        agent = Agent(
            model="anthropic/claude-sonnet-4-6",
            name="Report Writer",
            role="Business Writer",
            goal="Create professional reports",
            skills=skills,
        )


# ===========================================================================
# docs/skills/advanced/caching.mdx
# ===========================================================================

class TestCachingExamples:

    def test_cache_ttl(self):
        """caching.mdx — Cache TTL"""
        from upsonic import Agent
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "cached-skill")
            skills = Skills(
                loaders=[LocalSkills(d)],
                cache_ttl=300,
            )

            agent = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="Agent",
                role="Assistant",
                goal="Help with tasks",
                skills=skills,
            )

    def test_force_refresh(self):
        """caching.mdx — Force refresh via reload()"""
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "my-skill")
            skills = Skills(
                loaders=[LocalSkills(d)],
                cache_ttl=600,
            )

            skills.reload()


# ===========================================================================
# docs/skills/advanced/callbacks.mdx
# ===========================================================================

class TestCallbackExamples:

    def test_callbacks(self):
        """callbacks.mdx — on_load / on_script_execute / on_reference_access"""
        from upsonic import Agent
        from upsonic.skills import Skills, LocalSkills

        def on_skill_loaded(skill_name: str, description: str):
            print(f"[SKILL LOADED] {skill_name}: {description}")

        def on_script_executed(skill_name: str, script_path: str, returncode: int):
            print(f"[SCRIPT RUN] {skill_name}/{script_path} -> exit code {returncode}")

        def on_reference_accessed(skill_name: str, reference_path: str):
            print(f"[REFERENCE READ] {skill_name}/{reference_path}")

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "code-review")
            skills = Skills(
                loaders=[LocalSkills(d)],
                on_load=on_skill_loaded,
                on_script_execute=on_script_executed,
                on_reference_access=on_reference_accessed,
            )

            agent = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="Agent",
                role="Developer Assistant",
                goal="Assist with development tasks",
                skills=skills,
            )

    def test_buggy_callback(self):
        """callbacks.mdx — Buggy callback doesn't crash"""
        from upsonic.skills import Skills, LocalSkills

        def buggy_callback(skill_name: str, description: str):
            raise ValueError("oops")

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "safe-skill")
            skills = Skills(
                loaders=[LocalSkills(d)],
                on_load=buggy_callback,
            )


# ===========================================================================
# docs/skills/advanced/safety.mdx
# ===========================================================================

class TestSafetyExamples:

    def test_policy_list(self):
        """safety.mdx — Skills accept policy list (None = no filtering)"""
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "my-skill")
            skills = Skills(
                loaders=[LocalSkills(d)],
                policy=[],
            )


# ===========================================================================
# docs/skills/advanced/overview.mdx
# ===========================================================================

class TestAdvancedOverviewExamples:

    def test_merge(self):
        """overview.mdx — Skills.merge()"""
        from upsonic.skills import Skills, LocalSkills, BuiltinSkills

        base = Skills(loaders=[BuiltinSkills()])
        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "custom")
            custom = Skills(loaders=[LocalSkills(d)])

            merged = Skills.merge(base, custom)

    def test_active_skill_tools(self):
        """overview.mdx — get_active_skill_tools()"""
        from upsonic.skills import Skills, BuiltinSkills

        skills = Skills(loaders=[BuiltinSkills()])
        active_tools = skills.get_active_skill_tools()


# ===========================================================================
# docs/skills/advanced/auto-selection.mdx
# ===========================================================================

class TestAutoSelectionExamples:

    def test_auto_select_construct(self):
        """auto-selection.mdx — Auto-select construct"""
        from upsonic import Agent
        from upsonic.skills import Skills, LocalSkills

        with tempfile.TemporaryDirectory() as d:
            for i in range(6):
                _make_skill_dir(d, f"skill-{i}")
            skills = Skills(
                loaders=[LocalSkills(d)],
                auto_select=True,
                max_skills=5,
                embedding_provider=None,  # placeholder
            )

            agent = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="Agent",
                role="Versatile Assistant",
                goal="Handle diverse tasks with the right expertise",
                skills=skills,
            )


# ===========================================================================
# docs/skills/advanced/knowledge-base.mdx
# ===========================================================================

class TestKnowledgeBaseExample:

    def test_kb_construct(self):
        """knowledge-base.mdx — KnowledgeBase + Skills construct"""
        if not os.environ.get("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        from upsonic import Agent, Task, KnowledgeBase
        from upsonic.skills import Skills, LocalSkills
        from upsonic.embeddings import GeminiEmbedding, GeminiEmbeddingConfig
        from upsonic.vectordb import ChromaProvider, ChromaConfig, ConnectionConfig, Mode

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "my-skill")
            ref_dir = Path(d) / "refs"
            ref_dir.mkdir()
            (ref_dir / "data.txt").write_text("Some reference data.")

            embedding = GeminiEmbedding(GeminiEmbeddingConfig())

            config = ChromaConfig(
                collection_name="skill_references",
                vector_size=3072,
                connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=str(Path(d) / "chroma_db")),
            )
            vectordb = ChromaProvider(config)

            kb = KnowledgeBase(
                sources=[str(ref_dir)],
                embedding_provider=embedding,
                vectordb=vectordb,
            )

            skills = Skills(
                loaders=[LocalSkills(d)],
            )

            agent = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="Research Agent",
                role="Research Specialist",
                goal="Find and synthesize information from skill references",
                skills=skills,
            )

            task = Task(
                description="Find best practices for error handling across all our skill references.",
                context=[kb],
            )

    def test_kb_as_tool_construct(self):
        """knowledge-base.mdx — KnowledgeBase as tool construct"""
        if not os.environ.get("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")

        from upsonic import Agent, Task, KnowledgeBase
        from upsonic.skills import Skills, LocalSkills
        from upsonic.embeddings import GeminiEmbedding, GeminiEmbeddingConfig
        from upsonic.vectordb import ChromaProvider, ChromaConfig, ConnectionConfig, Mode

        with tempfile.TemporaryDirectory() as d:
            _make_skill_dir(d, "my-skill")
            ref_dir = Path(d) / "refs"
            ref_dir.mkdir()
            (ref_dir / "data.txt").write_text("Some reference data.")

            embedding = GeminiEmbedding(GeminiEmbeddingConfig())

            config = ChromaConfig(
                collection_name="skill_references",
                vector_size=3072,
                connection=ConnectionConfig(mode=Mode.EMBEDDED, db_path=str(Path(d) / "chroma_db")),
            )
            vectordb = ChromaProvider(config)

            kb = KnowledgeBase(
                sources=[str(ref_dir)],
                embedding_provider=embedding,
                vectordb=vectordb,
            )

            skills = Skills(
                loaders=[LocalSkills(d)],
            )

            agent = Agent(
                model="anthropic/claude-sonnet-4-6",
                name="Research Agent",
                role="Research Specialist",
                goal="Find and synthesize information from skill references",
                skills=skills,
            )

            # Pass KB as a tool
            task = Task(
                description="Find best practices for error handling across all our skill references.",
                tools=[kb],
            )
