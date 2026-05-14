# Changelog

## 0.76.3

Released on 2026-05-12.

### Changes

- TECH-639: remove Celery from LLMManager ([#586](https://github.com/Upsonic/Upsonic/pull/586))
- Tech 1407 ([#582](https://github.com/Upsonic/Upsonic/pull/582))
- chore: make mcp and fastmcp optional ([#577](https://github.com/Upsonic/Upsonic/pull/577))
- docs: restructure README around autonomous agents ([#573](https://github.com/Upsonic/Upsonic/pull/573))
- feat(agent): add cost property ([#579](https://github.com/Upsonic/Upsonic/pull/579))
- feat: add asqav governance integration ([#564](https://github.com/Upsonic/Upsonic/pull/564))
- fix(telemetry): make Sentry strictly opt-in ([#583](https://github.com/Upsonic/Upsonic/pull/583))
- refactor(tools): split ToolProcessor into five collaborators ([#581](https://github.com/Upsonic/Upsonic/pull/581))
- standardize layout under src/upsonic/prebuilt  ([#580](https://github.com/Upsonic/Upsonic/pull/580))

### Contributors

- [@jagmarques](https://github.com/jagmarques)
- [@DoganK01](https://github.com/DoganK01)
- [@onuratakan](https://github.com/onuratakan)
- [@IremOztimur](https://github.com/IremOztimur)

## 0.76.2
Released on 2026-04-22.

- feat: current_data optional, agent infers data source from notebook (#575) (`4c26d41b`)
- chore: New Version 0.76.2 (#572) (`377a65e6`)
- feat: AppliedScientist accepts any research_source + auto inputs (#571) (`fad994d2`)

## 0.76.1
Released on 2026-04-22.

- chore: bump Docs submodule to 85f6571 (`c1f756c4`)
- chore: New Version 0.76.1 (`805a1d55`)
- feat: rename AppliedScientist research_paper to research_source (#570) (`c4c30349`)

## 0.76.0
Released on 2026-04-21.

- chore: New Version 0.76.0 (#569) (`15fb46db`)
- feat: added prebuilt agent, AppliedScientist (#568) (`65850886`)
- feat: add directories to .gitignore for autonomous workspace and example (`77ce3fa8`)
- docs: clarify format and field rules in SKILL.md for progress tracking (`90d2256c`)
- feat: add new skills and system prompt for applied scientist experiments (`87f65737`)

## 0.75.0
Released on 2026-04-14.

- chor: version 0.75.0 (`86643357`)
- refactor: rag modules and providers refactored (#566) (`90d69bd2`)
- fix: small fix on RAG side (`20fe2a40`)
- refactor: rag refactor, mcp fix, new storage table adding (`cdf03af1`)
- fix: set auotonmous agent print True as default (#565) (`5c855b1d`)
- fix: set auotonmous agent print True as default (`00b699dd`)

## 0.74.4
Released on 2026-04-02.

- fix: fix profanity policy implementation (`6d6e9cc6`)
- chor: new version (`19eb2ac3`)
- refactor: set 'query_knowledge_base' True as default (`1d31687d`)
- fix: fix knowledgebase unit test for 'query_knowledge_base' (`17a787a0`)
- refactor: promptlayer threading fix, memory refactor (`dc5aa2d4`)
- feat: mail interface (`8f6b3786`)
- fix: prevent command injection in check_command_exists (CWE-78) (`110c633a`)

## 0.74.3
Released on 2026-03-30.

- refactor: user policies refactor (`5905d6c5`)
- fix: add threading for initializing ocr engines preventing multiple download (`4dfca6c9`)

## 0.74.2
Released on 2026-03-24.

- fix: unit test fix (`7c8e39d4`)
- fix: fix api key dependet unit test (`2af25d04`)
- chor: new version 0.74.2 (`5a7c3e2c`)
- refactor: steps refactor, time tracking fix, skills safety policy refactor (`54b60dd8`)
- fix: fix total duration time tracking (`88388772`)
- refactor: metric setting fix, safety policies for skills (`c49db1ec`)

## 0.74.1
Released on 2026-03-18.

- fix: fix failed smoke test (`6203cb7c`)
- fix: fix tests for skill integration (`736bb3ff`)

## 0.74.0
Released on 2026-03-18.

- chor: version 0.74.0 (`43df94c2`)
- fix: fix mcp smoke test cases (`e3c5cdc8`)
- feat: discord interface, skills integration (`9e74474f`)
- fix: MCP call_tool fails for streamable-http and SSE transports (`66f181a1`)
- feat: parameter support for apify tools (`de169d66`)
- vectordb: sanitize table and schema names in PgVector provider (`b35401b2`)

## 0.73.2
Released on 2026-03-13.

- fix: apify tool function schema fix (`274ecb2d`)

## 0.73.1
Released on 2026-03-12.

- chor: version 0.73.1 (`681e2906`)
- fix: fix unit tests (`891ea90b`)
- refactor: langfuse/promptlayer integration refactor (`362866ca`)
- doc: add IDE integration section to README (#544) (`2ae896c7`)
- feat: clanker type alias (`5b640da2`)

## 0.73.0
Released on 2026-03-09.

- refactor: bug fix, telegram hitl, new HITL features (`3823ff0f`)
- Add Discord community badge and CTA to README (#543) (`35482a5a`)
- feat: new model providers (`2db6a34a`)
- fix: fix failed otel test (`a7e6e253`)
- fix: fix stuck unit test (`b6504a31`)
- fix: fix pytest.ini (`ba2ca18d`)
- fix: fix tests.yml (`29c371f2`)
- fix: fix unit tests in github (`de7635c7`)
- fix: fix unit test run error (`33f03d2f`)
- fix: fix github workflow (`cf6416c6`)
- fix: fix unit tests (`af1ff3d9`)
- refactor: refactor usage tracking (`ba49f02a`)
- refactor: promptlayer integration, ToolKit refactor, some bug fix (`aeea80b1`)
- feat: add Examples repository as a git submodule (`8c824be9`)
- fix: env variable fix for bedrock (`a92a9fee`)
- refactor: small comment changes (`042f6dc9`)
- feat: OTel/Langfuse/Crawlee integration, model api updates, context manager fix (`8ea942fb`)
- feat: enhanced input type for anthropic, Team printing logic same as Agent (`634f676b`)
- feat: new model setting classes for models (`78946cd2`)
- docs: update README content and examples (#535) (`2dbedb30`)

## 0.72.6
Released on 2026-02-23.

- chor: version 0.72.6 (`5ff9148b`)
- fix: small fix on qdrant side, enhance qdrant smoke test (`0c377729`)

## 0.72.5
Released on 2026-02-21.

- chor: version 0.72.5 (`4468f302`)
- fix: fix call manager unit test (`db89d468`)
- refactor: handle finish_reason length, tool managers seperated, more max tokens for anthropic, handle pydantic response format in streaming (`055d38cd`)
- feat: supermemory storage, agent metrics (`cc90b22e`)

## 0.72.4
Released on 2026-02-19.

- chor: version 0.72.4 (`a34df241`)
- fix: fix wrong api usage and anthropic api usage for new versions (#533) (`a05e99b7`)
- fix: fix firecrawl smoke test usage (`2e239b96`)
- fix: fix unit test for firecrawl (`e9fbc907`)
- fix: fix wrong api usage and anthropic api usage for new versions (`b63bb995`)

## 0.72.3
Released on 2026-02-17.

- chor: version 0.72.3 (`e7aa097f`)
- refactor: package management refactor (`9fb629cd`)
- fix: fix dependency conflict between cloudpickle and snowflake-snowpark-python (`61f008bf`)
- fix: fix mem0 storage implementation and enhance tests (`ec0924b1`)
- fix: fix firecrawl tool attributes (`f5255767`)
- fix: fix unit test usage (`3bbe888d`)
- refactor: pr testing and firecrawl tool support (`77674c54`)
- feat: add token usage recording for tasks in Agent (`717c06ad`)
- feat: implement persistent background event loop for async execution in Agent (`f57949d4`)
- feat: add timeout and partial result handling in Agent execution (`13940a69`)

## 0.72.2
Released on 2026-02-14.

- fix: fix retry and task error status collision (`2d8b2452`)
- chor: version 0.72.2 (`26989d48`)
- fix: fix unit test for requiring openai api key (`85ed4544`)
- refactor: refactor team and its smoke tests (`76f1cfe3`)
- fix: fix smoke tests by using new printing usage (`012878ea`)
- refactor: wrap system prompt in AutonomousAgent tag (`1455a204`)
- refactor: smoke test for autonomous agent tools (`e4bcab74`)
- fix: fix greeting logic in interfaces (`29aa168c`)
- fix: change Agents.md to AGENTS.md for workflow set. fix greeting logic for interfaces (`ec238f4b`)
- fix: fix lazy importing of storage classes (`94bc1d83`)
- feat: heartbeat feature for AutonomousAgent in interfaces (`f21cd614`)
- chor: add 'fastmcp' as dependency (`1c5915b1`)
- fix: fix package management and test github workflow (`f558fe41`)
- fix: fix test workflow (`8af36a48`)
- fix: fix package management and unit test workflow (`ea2b6bcc`)
- feat: Team/interface streaming mode (`2a2b9b7b`)
- fix: fix test workflow (`72cc1fb9`)
- fix: fix package management (`dfca7045`)
- feat: Agent/Team as mcp tool, Team can be put inside Team (`e6299e4e`)

## 0.72.1
Released on 2026-02-12.

- chor: version 0.72.1 (`109e91c6`)
- chore: update package versions and refine Python version markers in dependencies (`e800b345`)
- chore: replace pdf2image with pymupdf in dependencies and enhance OCR timeout handling (`5382534d`)
- chore: update package dependencies with version markers for Python compatibility (`bc6ee3d6`)
- feat: implement asynchronous OCR processing and add new document conversion layer (`53755db1`)
- chore: update package versions for chromadb, docling-core, and fastapi (`72ff0fe9`)
- fix: fix unit tests (`fec6ec77`)
- fix: fix LLM usage tracking (`53125e16`)
- chore: specify uv req version (`eeacd201`)
- chore: update vulnerable packages (`effa05e9`)

## 0.72.0
Released on 2026-02-09.

- refactor: cli and interface compatibility (`6270c577`)
- refactor: remove data privacy policies and related tests from safety engine (`c9e58b9a`)
- refactor: added optional model parameter for context manager middleware (`a2e9d356`)
- docs: translate benchmarks QUICKSTART.md to English (`f66145a9`)
- chor: version 0.72.0 (`e93f2525`)
- feat: workspace feature, autonomous agent (`855053fc`)
- feat: telegram interface, chat/task modes for interfaces (`f056626e`)

## 0.71.6
Released on 2026-02-06.

- chore: update version to 0.71.6 in pyproject.toml and uv.lock (`87f8af98`)
- feat: implement de-anonymization in CallManagementStep for improved output clarity (`3204adc8`)

## 0.71.5
Released on 2026-02-05.

- docs: fix test command in CONTRIBUTING.md to use uv run (#520) (`69b54a09`)
- chore: update version to 0.71.5 in pyproject.toml (`46091987`)
- feat: add anonymization debug panel for enhanced logging (`a38ea738`)
- fix: Resolved anonymization action logic (#519) (`5c60ede6`)
- test: update mock async method to include print method control (`0f0091f2`)
- feat: enhance multi-agent execution with print method control (`38331e0b`)
- test: add unit test for bocha web search tool (`feb53b8d`)
- add bocha web search tool (`48644ddf`)
- feat: add comprehensive benchmark system for performance analysis (#511) (`78eada81`)

## 0.71.4
Released on 2026-01-31.

- chor: version 0.71.4 (`408905b7`)
- tests for new streaming events for chat class (`7a745819`)
- refactor: add event streaming for chat class (`8334df59`)
- fix: fix state mutuation race condition and inconsistent default behavior (`70e97409`)
- fix: fix memory error for streaming (`edbc6f2c`)
- refactor: refactor agent printing hierarchy (`0919fe42`)

## 0.71.3
Released on 2026-01-30.

- fix: fix unit tests (`69d6d609`)
- refactor: remove emoji (`92ba6df4`)
- chor: version 0.71.3 (`8bce0b2f`)
- refactor: adding original description of Culture into the prompt (`9e0cc73f`)

## 0.71.2
Released on 2026-01-29.

- chor: add Python language classifiers into pyrpoject.toml (`47ce7861`)
- fix: fix Agent printing (`3087e9b8`)

## 0.71.1
Released on 2026-01-29.

- refactor: 'print' flag for log printings (`fb8a8828`)
- fix: printing now depedent on fully 'debug' flag (`f85d062d`)
- refactor: refactor culture (`0bcbb862`)
- docs: improve developer onboarding flow and add infrastructure section (#509) (`44901e8f`)

## 0.71.0
Released on 2026-01-28.

- chor: version 0.71.0 (`75162a19`)
- fix: dependency migration, faiss/pgvector vectordb smoke tests, task metric fix when its Direct class (`19d1a3d4`)
- fix: fix milvus unittest (`fc5279cf`)
- refactor: contributing.md file and dependency grouping (`c301b979`)
- refactor: refactor interface, new  vectordb smoke tests (`3e8f7d31`)
- docs: refactor README to follow open-source standards (`df52504e`)
- refactor: easy import for Simulation (`6e7ff893`)
- feat: Simulation feature (`c55c4965`)
- fix: fix the unit test v2 (`c68a8998`)
- fix: fix unit test (`05f365bd`)
- refactor: culture logic refactor (`ae571cf6`)
- fix: fix unit tests (`22bfeaee`)
- fix: fix unit test (`33215c33`)
- feat: Cultural Knowledge (`04da1401`)
- fix: dependency fix (`bf7e28f7`)
- refactor: Makefile for smoke tests (`f988aea6`)

## 0.70.0
Released on 2026-01-17.

- chor: version 0.70.0 (`d30743c7`)
- refactor: pytest.ini update (`aebb9107`)
- fix: fix test for new architectures (`5db686e8`)
- refactor: storage providers/memory class refactor (`ad636b31`)
- fix: notebook fixed (`aff5197d`)
- refactor: remove AgentRunContext and make it all dependet on AgentRunOutput (`6135b416`)
- fix: test fix for python versions (`aee828db`)
- fix: fix milvus test stucking (`70c6b2dd`)
- fix: fix unit tests for 3.12 (`f8c8cdb8`)
- fix: moved smoke tests from unit_tests into smoke_tests (`1f94bfaa`)
- fix: fix tests getting stuck (`b1977cd5`)
- fix: remove old version files (`afd9071f`)
- fix: test fix, duplicate file remove (`363a8be2`)
- fix: system prompt setting up fix (`dc9a9d05`)
- Agent Run Implementation (`a3d0e62c`)

## 0.69.3
Released on 2025-12-25.

- chor: version 0.69.3 (`a813b00c`)
- fix: fix empty system prompt (`40c4ba0a`)
- Doc  new readme (#501) (`49e25669`)

## 0.69.2
Released on 2025-12-18.

- chor: version 0.69.2 (`a6218ec9`)
- fix: fix nested f strings formatting for python 3.11 (`3f5c5ebe`)

## 0.69.1
Released on 2025-12-16.

- fix: fix function calling parameters (`a60d3495`)
- fix: fix normalizing model ids (`cd8b15ab`)

## 0.69.0
Released on 2025-12-16.

- fix: add some missings (`763a1b4e`)
- fix: llmlingua prompt compression api fix (`1e4bbde9`)
- feat: version 0.69.0 (`2e0881dd`)

## 0.68.3
Released on 2025-12-12.

- chor: version 0.68.3 (`933ce2e9`)
- fix: fix bedrock provider (`3f28aa01`)

## 0.68.2
Released on 2025-12-12.

- chor: version 0.68.2 (`cd398e22`)
- fix: bedroc profile creation fix (`c167f1a3`)
- chore: image output utils (`c281715b`)
- chor: add cloudpickle as dependency (`de779ce9`)
- refactor: Refactor durable feature, new model classes, fix 'pgvector required' (`c1942005`)
- refactor: new smoke tests and bug fix (`755b18b3`)
- fix: update OCR prompt template in DeepSeekOllamaOCR class (`9f19b05a`)
- feat: Added timeout mechanism to deepseec-ocr ollama (`89dd752d`)
- feat: Added deepseek-ocr with ollama (`782a452d`)

## 0.68.1
Released on 2025-12-04.

- chor: version 0.68.1 (`ee05aca5`)
- fix: fix docstring of search tool from knowledgebase, mcp closing method adding added via Task class (`1d30e17e`)

## 0.68.0
Released on 2025-12-02.

- fix: tool unit test fix (`c58c920e`)
- fix: bug fixing  and rag as tool (`bd5f5c4a`)
- fix: fix unit tests (`46d48031`)
- fix: conftest.py fix (`3bdeb6ae`)
- feat: interfaces added (whatsapp, gmail and slack) (`c17cafeb`)
- feat: KnowledgeBase as tool feature added (`f52add7a`)
- feat: DeepAgent refactored and tool adding via Agent class added (`83252024`)
- feat: Safety policy for tools added (`8e13503f`)
- refactor: import log method added for import errors and groq import error fixed (`a1b5aa19`)
- refactor: refactor test files for changes (`5e3da381`)

## 0.67.4
Released on 2025-11-27.

- chor: new version 0.67.4 (`3f4ab914`)
- fix: fix run command not importing other files (`4499fe86`)

## 0.67.3
Released on 2025-11-25.

- refactor: new version 0.67.3 (`3e88faca`)
- fix: json config data fix in 'upsonic init' command (`8c8195ef`)
- feat: zip command (`05a2ad1d`)
- refactor: import error log adding and circular import fix (`ea28804e`)

## 0.67.2
Released on 2025-11-25.

- new version 0.67.2 (`edad152f`)
- fix: unit test fixes (`c2f956da`)
- fix: fix unit tests (`13e301cb`)
- fix: correct YFinanceTools import and update notebook cells (`f669f12e`)
- tests : unit tests added for the tools feauture (`526307a6`)
-  unit tests added for the graph feature (`c99df9dd`)
-  unit tests added for team feature (`fddd9925`)
- doc: Update introduction to clarify target users (`11b1e6b1`)
- policy manager test fix (`c23075c3`)
- fix of unit tests commit (`92a3d348`)
- cli update for agentos (`5dfbb6f6`)
- fix: dependency conflict fix (`b96fd10c`)
- refactor: mcp tool handling class refactor (`e2c163e1`)
-  unit tests added for team feature (`e28522b6`)
-  unit tests added for the graph feature (`d72607cd`)
- tests : unit tests added for the tools feauture (`e3334342`)
- fix: correct YFinanceTools import and update notebook cells (`4f84b56a`)

## 0.67.1
Released on 2025-11-18.

- chore: v0.67.1 (`23b273d9`)
- fix: fix typing issues (`3ea957e6`)
- fix: tool context parameter removing (`25604935`)
- feat: interface, tool handling refactor (`9f3dc898`)
- refactor: image output support adding and general refactor (`f0ff05a8`)

## 0.67.0
Released on 2025-11-13.

- fix: vectordb init fix (`495d5e20`)
- feat: 67.0 release (`aaa0ed76`)
- refactor: remove 'run' optional dependency group (`8faa83e9`)
- doc: Update introduction to clarify target users (`3ecc6bd4`)
- refactor: lazy import, command dispatch (`11c964f9`)
- policy manager test fix (`0d2ab443`)
- refactor: change 0.0.0.0 to localhost (`b8a8cf84`)
- feat: install command and fasten cli (`9e4deeea`)
- fix of unit tests commit (`ca36cb6f`)
- feat: CLI support for upsonic (`2696c90d`)
- UEL refactor and output parser adding (`00432373`)
- fix: unit tests fixing (`bacdaa91`)
- fix: reuploading uv.lock (`23d5a787`)
- chore: local changes to pyproject and uv.lock (`61966795`)
- fix: fix test files (`b0b15c5f`)
- fix: fix ContextManager class based on the changes on KnowledgeBase class (`c6c56c4d`)
- refactor: vectordb v2 (`a6e7a894`)

## 0.66.1
Released on 2025-11-06.

- chore: update version to 0.66.1 (`6da3fd24`)
- fix: Problem resolved on ollama usage with model as string (`9aeae511`)

## 0.66.0
Released on 2025-11-03.

- fix: version fix (`cc93984a`)
- fix: fix unit test failings (`f4f0bb05`)
- fix: logging fix, memory error handling adding (`bbaf3247`)
- fix: fix unit tests (`4e14a2ea`)
- feat: OCR Enginer custom model support (`e058171a`)
- fix: fix some bugs and external tool call (`eab5e038`)
- feat: add printings for ocr modules and missing packages (`987b541f`)
- fix: fix dependency conflict (`39c69c35`)
- feat: durable execution, ocr engine (`22cf7507`)
- docs: fix broken links in README (`cb9994ec`)
- feat: Direct class for fast and simple model request (`ba0fcd03`)
- feat: state graph, recursive tool handling for streaming logic (`b6ffaee3`)
- smoke test for mcp added (`039a8c66`)
- fix: test fix (`b9965e87`)
- feat: multiple policy handling (`218854ce`)
- fix: fix price not found error (`8567e311`)
- fix: test fixes (`961262b5`)
- feat: Expression Language for Upsonic (`5e21ddc2`)

## 0.65.1
Released on 2025-10-20.

- chore: update version to 0.65.1 in pyproject.toml and uv.lock (`2250d0e2`)
- feat: Enhance telemetry and logging features with comprehensive documentation and tests (`a1e5947a`)
- fix: Update Sentry logging integration to capture INFO+ level logs and remove unnecessary breadcrumbs (`ffe1be87`)
- fix: New Tracing and centralized logging system (`30b08b67`)
- fix: Resolved logging problem (`ca28976a`)

## 0.65.0
Released on 2025-10-17.

- fix: update version to 0.65.0 in pyproject.toml and uv.lock (`ade30622`)
- fix: Resolved the deprecation warnings (`380957d1`)
- fix: Resolved most of the errors of smoke tests (`f82dd898`)
- fix: add 'gemini' provider support in model inference (`d7f9ada9`)
- fix: company name and other company releated attributes on agent object fixed (`640d6d13`)
- fix: fixing files for unittests (`f55fa95d`)
- refactor: refactor deep agent todo handling (`5579c33f`)
- fix: fix stream logic (`4c05bcfa`)
- fix: chat streaming fix (`cfe8893a`)
- feat: reasoning and thinking attributes mapping (`66ca2c56`)
- smoke suit 2 finished (`b71f2e67`)
- feat: add task_start method to MockTask class for improved testing (`ba6bd54c`)
- feat: enhance task management and pipeline logging (`fd83515a`)
- feat: add temp directory to .gitignore (`e567d507`)
- feat: deep agent and pipeline (`78d41573`)

## 0.64.1
Released on 2025-10-13.

- refactor: Changed location of smoke test (`a5ae25e5`)
- refactor: MongoStorage added to init (`da619eb2`)
- All smoke test suit (`87dbb76b`)
- Added team, agent and model smoke tests (`829cd4ea`)
- fix: Update project version to 0.64.1 in pyproject.toml and uv.lock (`22bf9da4`)
- refactor: All import system improved (`edd38875`)
- fix: Update project version to 0.63.1 (`3239b336`)
- fix: Improve smoke tests by adding timeout and simplifying task configurations (`777fdeb6`)
- reformat: Test files reorganized (`c53671ea`)
- tests: Add smoke test for task functionality (#441) (`dadf253a`)
- fix: Backward compatiblity for LLM_MODEL_KEY and bypass_llm_model (`143937ff`)
- fix: Backward compatibility for Azure OpenAI (`d0d65e28`)
- fixed a tpyo after Pull Request Review (`be6ef40b`)
- Add smoke test for task functionality (`668eb304`)

## 0.64.0
Released on 2025-10-12.

- feat: add mem0ai dependency and update protobuf version (`0e902c55`)
- chore: Sync with master (`3b91eb3f`)
- refactor: fix tests (`d853855f`)
- feat: add mem0ai dependency and update protobuf version (`1e75d737`)
- chore: Sync with master (`e98841b8`)
- feat: mem0 integration, refactor attachment attribute (`a44e16cd`)

## 0.63.0
Released on 2025-10-11.

- chore: update version to 0.63.0 in pyproject.toml (`85f93011`)
- feat: pdfplumber loader and its tests (`0cb851ae`)
- fix: dependency fix for unittests (`c39cca07`)
- refactor: fix not awaited warning (`db353f5e`)
- tests: new tests for vectordbs (`8ba59395`)
- fix: update import paths to remove 'src' prefix in test_embedding_providers.py (`0e552b2b`)
- feat: docling loader, new tests (`658859f7`)
- feat: Auto model, console printing  adding, new policies (`fb3f85e3`)

## 0.62.0
Released on 2025-10-07.

- fix: update dependency installation to use all extras (`fcb786e9`)
- release: 0.62.0 (`054f67d4`)
- fix: remove fixed width from import error table (`3f7d9672`)
- feat: Prints improved for import errors (`89a4c8b9`)
- feat: chat class, package grouping, import error handling (`473a49a1`)
- feat: tests for new policies (`ab0d2152`)
- fix: fix base64 method (`b4d6dd7f`)
- fix: fix tests and task (`0b436593`)
- feat: new policies, builtin tool handling, bug fixes (`c075faef`)
- feat: adding load_dotenv (`f83b36c6`)
- feat: test file for pymupdf (`584b22e2`)
- feat: pymupdf loader (`807b1394`)
- refactor: refactor utils, add built-in web search/read 'functions' (`1f332990`)
- refactor: tests are fixed (`fe14695c`)
- refactor: Lazy import and Memory fix (`21ed79e8`)
- refactor: New Agent Class (`d18345c5`)
- refactor: deleting unused config parameter (`0027357c`)
- refactor: enhancement for csv loader (`e453fcb1`)
- fix: fix unit tests (`22b7858b`)
- refactor: refactor test files for new loaders and chunkers (`22e46f72`)
- fix: fix proper logic for custom api and model (`c8aa7385`)
- refactor: refactor for custom api and model support (`9a9810b2`)
- fix: Fixing vectordbs (`cd5120c1`)
- fix: Small fixes for thinking-reasoning tool feature (`60405e30`)
- refactor: tiny fixes and useful methods (`245b3172`)
- docs: Update guides section header to reflect the correct number of steps (`a2ac5379`)
- docs: Revise README to reflect new Upsonic framework features and focus on AI agent development for fintech (`0b695cbf`)

## 0.61.1
Released on 2025-09-11.

- chore: Bump version to 0.61.1 and update upload times in dependencies in pyproject.toml and uv.lock (`c950f92b`)
- refactor: Remove debug print statement from manage_context method in ContextManager (`febecea2`)
- refactor: Remove comprehensive test files and add simplified test files for various loaders and chunkers (`2dbcc28b`)
- ci: Update Python version matrix in GitHub Actions workflow to remove 3.9 (`fb88bdcd`)
- test: Enhance unit tests with async mocking for agent functionality and context handling (`96cdc3e4`)
- refactor: Changes on laoders, splitters rag pipeline, new feature for vectordb (`b89c0812`)
- feat: Add GitHub Actions workflow for running unit tests across multiple Python versions (`5528810e`)
- docs: Update CLAUDE.md to reflect changes in example usage and troubleshooting instructions (`c459d256`)
- feat: Add yfinance and its dependencies to project configuration (`3c20185d`)
- docs: Add CLAUDE.md for project overview, architecture, development commands, and testing structure (`8e9af375`)
- "Claude Code Review workflow" (`e953511b`)
- "Claude PR Assistant workflow" (`34460f60`)
- feat: Introduce custom exceptions for improved error handling for missing api keys (`963692fe`)
- refactor: Remove upsonic_error_handler decorator from various classes (`58fff7b4`)
- feat: Redesigned info prints (`2ce6e6b4`)
- fix: Tool import problem resolved (`f584c69c`)
- feat: New WebSearch and WebRead tools added (`025c1c0c`)
- fix: Restore necessary print statements for debugging in agent and embedding factory (`af9a6582`)
- Refactor: Remove debug prints and console logs (`06850227`)
- fix: Set default model to "openai/gpt-4o" in Direct agent initialization (`ba4bcb6f`)

## 0.61.0
Released on 2025-09-10.

- chore: Version Updated (`c53435ee`)
- fix: Resolved safety engine prebuilt policies (`001319d2`)
- test: Add comprehensive unit tests for TextLoader functionality (`fe4d88a9`)
- test: Add comprehensive unit tests for AgenticChunkingStrategy (`90fad877`)
- tests: Added rag/chunking and rag/loaders unit tests (`9e0a2232`)
- feat: cache feature for prompts (`f5573a12`)
- refactor: remove unnecessary comments (`c8b02e6c`)
- feat: external tool call (`add74dcd`)
- feat: HF remove local cache (`676861a1`)
- fix: fixing unncessary close method in HF (`5086f476`)
- refactor: Closing feature, intelligence init and some fixing (`36dfa1da`)
- refactor: Rule chunking method removed (`07df189e`)
- feat: Refactor on loaders (`8a7ebd74`)
- feat: safety_engine (`193b1518`)
- tests: Added tests for the old model system support (`43129aea`)
- refactor: Change location of rag dependencies into the optional 'rag' (`deecc9f7`)
- refactor: Refactor RAG features (`26cbbc01`)
- refactor: Refactor  Model and Provider initialization with string (`b99c9dce`)
- refactor: Refactor context manager and gemini embedding provider (`b7788b50`)
- feat: RAG (`59f6bf12`)
- doc: Changed banner (`a39c12a3`)
- doc: Changed banner (`7d26e606`)
- doc: Banner changed (`99c7d5f5`)
- fix: refine guardrail error handling in retry logic (`aa9075e3`)
- fix: guardrail error handling fixed (`384209e2`)
- fix: fix guardrail retry v2 (`8e18c857`)
- feat: Optional memory attribute for coordinate mode in Team class (`d10a50e7`)
- fix: guardrail retry fix (`be8998f3`)
- feat: Guardrail feature for Task class (`3b32a377`)
- feat: Eval layer and new model classes (`aacbfaa9`)
- feat: Thinking and Reasoning Tool (#418) (`e401e754`)
- feat: Reasoning support (#415) (`dbcae1df`)
- feat: Added MongoDB storage (#414) (`c2f1ba55`)
- feat: New memory and tool features: (#413) (`e574000e`)
- feat: implement hybrid sync/async memory and storage system (#412) (`f0e566b8`)
- feat: add new model 'ollama/gpt-oss:20b' to model registry (`e6bcc2af`)
- chore: add video.mp4 to .gitignore (`30bcc021`)
- refactor: update task assignment to use attachments and default LLM model (`19db1694`)
- refactor: update Task model to use None for default values of tools, context, and _tool_calls (`2412b12c`)
- refactor: deleting a print statement (`a29bc006`)
- refactor and feat: refactor tool handling, add image, audio and video as input, refactor context preparing in team , add agent as tool in team. change behavioral_wrapper from sync to async and calling logic of it (`428a2470`)
- refactor: Remove debug print statements from processor, tool_usage, and test files (#409) (`62b69353`)
- tests : convert agent tool function call test to unittest framework (`399131ed`)
- feat: Added ollama models (`90d5469b`)
- tests : fixed  test_agent_tool_function_call test by making it pytest based (`3f4e53c6`)
- feat: new feautures for system prompt (`0edf4fa4`)
- docstring for new decorator (`097eb02b`)
- feat: agent as tool and retry decorator (`154f0b7f`)
- tests : Added  test_agent_tool_function_call test (`48d8001b`)
- Storage and Tool Improvements (#404) (`e8f71166`)
- tests: Added initial test cases for task and agent releation (`e2c9b04a`)
- refactor: Update default prompt to clarify agent's role and task completion capabilities (`b7769cda`)
- context.py deleted, system prompt manager refactored a little (`e00c45a0`)
- feat: Refactor graph, new features and refactor for context management (`986a29af`)
- Revert "feat: Add Core Built-In Tool Capabilities to the Upsonic Runtime (#391)" (#394) (`fc3d7dd9`)
- feat: Add Core Built-In Tool Capabilities to the Upsonic Runtime (#391) (`d40def3d`)
- feat: accept string as context (#378) (`bb7eeda5`)
- refactor: Update imports and remove ObjectResponse class, replacing it with BaseModel in relevant files (`7ecfc9a4`)
- refactor: Remove MemoryManager class and integrate memory handling directly into memory.py for streamlined functionality (`cbf0e539`)
- refactor: Remove handler classes and streamline context managers for improved clarity and efficiency (`1ffa3134`)
- refactor: Fixed all false imports (`0600bef2`)
- refactor: Migrate Direct folder to agent (`1ec1604a`)
- refactor: Consolidate agent creation and tool registration into Direct class, removing redundant files (`1042c222`)
- refactor: Created context managers for direct (`b2a6790d`)
- refactor: Move agent input building logic to Task class and streamline async execution in Direct class (`29622961`)
- refactor: Simplify task execution flow by removing nested function and directly handling task responses (`9411ee2c`)
- refactor: Update agent creation function to use llm_model directly and retrieve agent model within the function (`3ab7294b`)
- refactor: Moved direct llm call utils to direct llm call folder (`aa4c2e19`)
- refactor: Task end response and start functions moved tot ask class itself (`160afa0a`)
- feat: Added example Investment Report Generator agent (#382) (`effa7811`)

## 0.60.0
Released on 2025-07-02.

- chore: Bump version to 0.60.0 in pyproject.toml and uv.lock (`bd6570ee`)
- feat: Manual task assignment in team (#375) (`ee4cf7e4`)
- feat: Integrate reliability layer into Direct class and update Task context initialization (`fa1ee26c`)

## 0.59.11
Released on 2025-07-01.

- chore: Bump version to 0.59.11 in pyproject.toml and uv.lock (`7dc2e405`)
- feat: Add initial test suite for Upsonic with basic tests for Task and Agent functionality (`9ce02b30`)

## 0.59.10
Released on 2025-07-01.

- chore: Bump version to 0.59.10 in pyproject.toml (`706c6c66`)
- refactor: Clean up model_set function by removing unnecessary print statements and error handling (`926b5df2`)

## 0.59.9
Released on 2025-06-29.

- feat: Bypass model options for platform (#374) (`5bf6cc0d`)

## 0.59.8
Released on 2025-06-25.

- chore: Bump version to 0.59.8 in pyproject.toml and uv.lock (`c01a8ace`)
- refactor: Remove debug print statement from llm_usage function in llm_usage.py (`749356ec`)

## 0.59.7
Released on 2025-06-24.

- chore: New version (`1155e156`)
- feat: Update llm_usage function to extract actual token counts from usage data (`c027536d`)

## 0.59.6
Released on 2025-06-24.

- chore: Bump version to 0.59.6 in pyproject.toml and uv.lock (`b6eb6501`)
- feat: Reset price_id and clear tool call history for tasks in direct_llm_cal.py (#370) (`a4f9bc53`)

## 0.59.5
Released on 2025-06-19.

- chore: Bump version to 0.59.5 in pyproject.toml and uv.lock (`87071d72`)
- fix: Correct media type determination for image files in direct_llm_cal.py (#369) (`74a87c87`)

## 0.59.4
Released on 2025-06-17.

- Fix for graph and mcp sse (#368) (`9937f3fd`)
- chore: Update .gitignore to include paper.png (`0fe8829a`)

## 0.59.3
Released on 2025-06-17.

- fix: Remove debug print statement from KnowledgeBase class in knowledge_base.py (#367) (`d502bf83`)

## 0.59.2
Released on 2025-06-17.

- chore: Bump version to 0.59.2 in pyproject.toml and uv.lock (`7bcea4bd`)
- feat: Enhance context processing to include Knowledge Base information in context output (#366) (`13ee7178`)

## 0.59.1
Released on 2025-06-17.

- chore: Bump version to 0.59.1 in pyproject.toml and uv.lock (`8d80f95b`)
- feat: Add DuckDuckGo search functionality in tools.py (#365) (`841a4a74`)

## 0.59.0
Released on 2025-06-17.

- chore: Bump version to 0.59.0 in pyproject.toml and uv.lock (`711d30a9`)
- feat: Added memory to agents (#364) (`85918246`)

## 0.58.0
Released on 2025-06-16.

- chore: Bump version to 0.58.0 in pyproject.toml (`6f6a6ab0`)
- feat: Added default prompt structure (#363) (`cc6fa0ce`)
- feat: Refactor multi-agent architecture by replacing MultiAgent with Team class and enhancing task context management (`955f5c35`)
- feat: Multi Agent Implemented and context procesing improved (#362) (`1ffbbf4c`)

## 0.57.0
Released on 2025-06-16.

- refactor: Remove unnecessary print statements in Canvas class during save operations (`94b3417d`)
- feat: Integrate Canvas functionality into Direct class and Task management (#361) (`b10444d5`)
- doc: Edits for readme for new version of framework (`be6e995f`)

## 0.56.1
Released on 2025-06-15.

- fix: Remove MCP tools from the tools list after processing in agent_create function (`d79c18d2`)
- fix: Update _display_error function to handle generic Exception types and improve error message retrieval (`58f747d8`)
- chore: Bump version to 0.56.1 in pyproject.toml and uv.lock; add reliability_layer parameter in Direct class (`373934fc`)

## 0.56.0
Released on 2025-06-15.

- chore: Bump version to 0.56.0 in pyproject.toml (`4ba3a843`)
- Stability (#360) (`a54529ac`)
- Update README.md (`f5dd5df1`)
- Update README.md (`5a93143c`)

## 0.55.6
Released on 2025-05-18.

- chore: Bump version to 0.55.6 in pyproject.toml and update model name for bedrock/claude-3-5-sonnet (`97f9f8fd`)

## 0.55.5
Released on 2025-05-16.

- chore: Bump version to 0.55.5 in pyproject.toml (`65cb007f`)
- refactor: Simplify scaling configuration for HD displays and remove unused resolutions (`36e1df20`)
- fix: Update VNC server geometry to 1280x720 in Dockerfile (`c183f25b`)

## 0.55.4
Released on 2025-05-05.

- chore: Bump version to 0.55.4 in pyproject.toml and uv.lock (`31aa9c9b`)
- fix: Add traceback logging for server start failure in ServerManager (#351) (`d68b2636`)

## 0.55.3
Released on 2025-05-05.

- chore: Bump version to 0.55.3 in pyproject.toml and uv.lock (`fce4105d`)
- fix: Ensure single worker for tool server in ServerManager (#350) (`1f00d6e4`)

## 0.55.2
Released on 2025-05-04.

- chore: Bump version to 0.55.2 in pyproject.toml and uv.lock (`a4917677`)
- feat: Add configurable worker count for server process in ServerManager (#349) (`da1a4d49`)

## 0.55.1
Released on 2025-05-02.

- chore: Bump version to 0.55.1 in uv.lock (`c69f4456`)
- fix: Resolved different OLLAMA_BASE_URL (#347) (`d5159ec2`)

## 0.55.0
Released on 2025-05-02.

- chore: Bump version to 0.55.0 in pyproject.toml (`912bde3a`)
- feat: Added tool_calls to the task class (#346) (`5a533028`)
- feat: Added tool call information to direct (#345) (`c5f35658`)

## 0.54.0
Released on 2025-05-01.

- chore: Bump version to 0.54.0 in pyproject.toml (`782e90ec`)
- feat: Add properties to Task class for total input and output tokens, and improve total cost calculation (#344) (`ab0bbe5e`)
- feat: Enhance call_end function to track price ID usage and estimated costs (#343) (`8ab10782`)

## 0.53.1
Released on 2025-04-26.

- fix: Fix for python 3.10 and pydantic situation (#341) (`c17e5e3f`)

## 0.53.0
Released on 2025-04-25.

- chore: Bump version to 0.53.0 in pyproject.toml (`620c6460`)
- feat: Added team feature (#339) (`9431ad50`)
- fix: Add checks for UV and Node.js installation in tool registration to prevent execution errors (#338) (`bb353591`)
- fix: Resolved crawl method in Crawl4AISimpleCrawling by removing the configuration on browser (`184908d6`)
- fix: Update FirecrawlScrapeWebsiteTool to unpack options when scraping URLs (`d4cf2950`)
- fix: Pass client parameter to Direct.do_async calls in Agent class for fixing client usage with agent characterization (`bcc219ef`)
- feat: Added support to BaseModel on response_format and removed custom responses: StrResponse, IntResponse, FloatResponse, BoolResponse, StrInListResponse (`17c7f8be`)
- chore: Update dependencies (`6cedb66b`)

## 0.52.4
Released on 2025-04-19.

- refactor: Remove asyncio dependency check from Crawl4AISimpleCrawling (`57c51f4c`)
- chore: Bump version to 0.52.4 in pyproject.toml (`2a2f281d`)
- fix: Turned Crawl4AISimpleCrawling to async directly (`0eba9f2a`)
- fix: Simplify return statements in ComputerTool actions for clarity (`8a0740be`)
- fix: Computer use approach seperated for different operating systems (`e7d05341`)

## 0.52.3
Released on 2025-04-09.

- chore: Bump version to 0.52.3 in pyproject.toml (`891cc3eb`)
- feat: Add Screenshot class to tools and update exports (`55b01421`)
- fix: Fixed screenshoot ability for computer use (`7e2ed886`)

## 0.52.2
Released on 2025-04-08.

- chore: Bump version to 0.52.2 in pyproject.toml (`a327d3af`)
- chore: Dependency update (`fa328c2e`)
- feat: Added async custom tool support (#329) (`c826ed94`)

## 0.52.1
Released on 2025-04-05.

- chore: Bump version to 0.52.1 in pyproject.toml (`de151883`)
- feat: Add retry parameter to various LLM call methods for improved error handling and resilience (`8bb4648b`)
- deprecation: Removed old reflection (`82897f1f`)

## 0.52.0
Released on 2025-04-04.

- chore: Bump version to 0.52.0 in pyproject.toml (`9f9d8ac4`)
- fix: Error handling for server (`68919ab6`)
- feat: Introduce UnsupportedComputerUseModelException and enhance tool compatibility checks for ComputerUse capabilities (#324) (`7afdb848`)
- feat: Added tool call printing feature (#322) (`a56e99de`)
- feat: Added intellisense support for model names (`ef307145`)
- feat: Load environment variables for Sentry configuration and set environment in telemetry (`f63d0a00`)
- fix: Enforce named parameter usage for Direct initialization and update documentation with examples (#320) (`57db7531`)

## 0.51.2
Released on 2025-04-03.

- chore: Bump version to 0.51.2 in pyproject.toml (`d5ff2fe0`)
- refactor: Update model settings to use specific model settings classes and add handling for non-parallel OpenAI models (`ef7dd47d`)

## 0.51.1
Released on 2025-04-03.

- chore: Bump version to 0.51.1 in pyproject.toml (`e88bdb9a`)
- fix: Resolved library version tracking for better debugging (`4b265675`)

## 0.51.0
Released on 2025-04-03.

- fix: Correct model identifier format for Ollama Llama 3.1 in model registry (`0aefe604`)
- chore: Bump version to 0.51.0 in pyproject.toml (`57556fde`)
- feat: Enhance model registry by adding required environment variables for API keys (`f9fe66d3`)
- fix: Handled an error for deepseek chat (`81c0e03e`)
- fix: Update default pricing for models to zero in model registry (`18586ce4`)
- feat: Add OpenRouter model support and API key configuration (`20953aee`)
- feat: Add new model definition for Ollama Qwen 2.5 in model registry (`f84ccb02`)
- feat: Add new model definitions (`c629e297`)
- feat: Implement Ollama model creation and integrate into model registry (`1cae1b9a`)
- feat: Add new model definition for Ollama Llama 3.2 in model registry (`dc0533ef`)
- fix: Update comments for clarity and enhance characterization logic for ollama models in Agent class (`0e5864d1`)
- feat: Add new model definition for Gemini 2.0 Flash in model registry (`629c4f96`)
- feat: Add support for Gemini model and Google GLA API key configuration (`9b1f5b2f`)
- feat: Add new model definition for OpenAI GPT-4.5 Preview in model registry (`cd2d8838`)
- feat: Add new model definition for Claude 3.7 Sonnet in model registry (`813746d8`)
- fix: Update Anthropic model initialization to use provider class for API key management (`8e84c98d`)
- Deprecated old model definations (`c00e7744`)
- feat: Introduce centralized model registry and pricing management in model_registry.py (`caaef0f2`)
- refactor: Enhance error handling and message preparation in CallManager and AgentManager (`0f4df853`)

## 0.50.5
Released on 2025-03-28.

- chore: Pin pydantic version to 2.10.4 in pyproject.toml and uv.lock (#315) (`0f433aff`)
- fix: Remove timeout decorators from Direct and Agent endpoints (#313) (`3bc42e37`)
- chore: Bump version to 0.50.5 in pyproject.toml (`c898ee64`)
- feat: Add environment variable support for BASE_PATH configuration (#311) (`7f339292`)

## 0.50.4
Released on 2025-03-27.

- chore: Bump version to 0.50.4 in pyproject.toml (`de08c0cf`)
- fix: Increase timeout duration for server startup and completion checks (`800b5a71`)
- fix: Enhance error handling and logging in server startup process (`a4d31079`)

## 0.50.3
Released on 2025-03-27.

- chore: Bump version to 0.50.3 in pyproject.toml (`5a99cc07`)
- fix: Added support to run with streamlit (#309) (`f4171cc6`)

## 0.50.2
Released on 2025-03-27.

- refactor: Update agent_creator to use provider parameter for OpenAI and Anthropic models (`e007119f`)
- chore: Bump version to 0.50.2 in pyproject.toml (`5cea9128`)
- chore: Update dependencies (`9262cb75`)

## 0.50.1
Released on 2025-03-26.

- chore: Bump version to 0.50.1 in pyproject.toml (`e33e5048`)
- refactor: Replace signal-based timeout handling with asyncio for improved performance and reliability in timeout decorator (`7176e54b`)

## 0.50.0
Released on 2025-03-25.

- feat: Update settings initialization to use ModelSettings for OpenAI and Anthropic configurations (`73d272ea`)
- feat: Add Canvas feature (#307) (`73aa96bf`)
- feat: Introduce tool_operation function for formatted tool operation output and integrate it into agent registration process (`b8163623`)
- feat: Add error_message function for formatted error handling and integrate it into error_handler (`c31a9754`)
- feat: Implement mcp_tool_operation function for formatted MCP tool operation output (`2447002c`)
- fix: Enhance AgentConfiguration and Task models with additional parameters and type annotations for improved flexibility (`c58addb3`)
- fix: Refactor AgentConfiguration and Task initialization to improve parameter handling and default values (`9d1f007d`)
- fix: Refactor message handling in AgentManager to utilize agent_memory for improved context management (`f506af22`)
- fix: Update README to reflect change in usage of web_agent.print_do (`23a9d742`)
- chore: Update dependencies and improve message handling in CallManager and AgentManager (`58a9d2d2`)
- fix: Add response_lang to Task initialization in level_no_step and end_task (`ad1ee733`)
- fix: Improve signal handling in ConfigManager to ensure graceful exit (`fbb1ab9f`)
- chore: Bump version to 0.50.0 in pyproject.toml (`4ce09b63`)
- feat: Added response_lang setting to Task (#306) (`c39ade0f`)
- feat: Add support for SSE MCPs  (#305) (`026265aa`)
- fix: Remove erroneous line (`76f9c8ae`)
- feat: Enhance error handling in API and server components (`fb4a7302`)
- feat: Enhance tool listing by truncating descriptions longer than 1024 characters to improve readability (#299) (`f57d58a4`)
- fix: Fixed object usage in returns for async functions (#298) (`38a22c5e`)
- Implement static methods in BrowserUse class for dependency analysis and import control (#297) (`23cc55b7`)
- Refactor Task class to simplify control method execution by removing unnecessary error handling (`2b372d8d`)
- Enhance 'rag' optional dependencies in pyproject.toml by adding 'future', 'pipmaster', 'tenacity', and 'tiktoken' for improved functionality (`60a8e65f`)
- Enhance optional dependencies in pyproject.toml by adding 'rag' group for improved modularity (`5329364f`)
- Refactor KnowledgeBase and Task classes to support asynchronous operations (`178b654c`)
- Refactor optional dependencies in pyproject.toml and uv.lock (`b2ab8ea4`)
- Update LightRAG logging function name. (#289) (`b59c5c29`)
- feat: Add error handling decorator to standardize server error responses in API endpoints (#295) (`f6399c04`)
- fix: Improve estimated cost calculation in agent_end function with error handling (#294) (`dd07fdc3`)
- feat: Escape rich markup in Graph and printing modules for safer text rendering (#292) (`223e313a`)
- Update issue templates (`675e13ce`)
- Update issue templates (`39596330`)

## 0.49.0
Released on 2025-03-15.

- chore: Bump version to 0.49.0 in pyproject.toml (`ad788e02`)
- feat: Add decision nodes (DecisionFunc and DecisionLLM) to Graph for enhanced execution flow control (`bd6a4dac`)
- feat: New Graph feature (`2cd5b9b9`)
- feat: Introduce Graph class for task management and execution, enabling task chaining and state management (`76088af1`)
- chore: Update logging level to ERROR in API modules for improved error visibility (`b249e461`)
- fix: Correct initialization of agent attribute in Task model for consistency (`ea0c3513`)
- feat: Add agent attribute to Task model for enhanced task management (`ae73794e`)

## 0.48.0
Released on 2025-03-14.

- chore: Bump version to 0.48.0 in pyproject.toml (`eaff9fee`)
- feat: Implement centralized exception handling with logging for improved error management in API (`61996297`)
- feat: Enhance task prompts by incorporating system prompt and updating agent name reference for improved context (`5ca2eb68`)
- feat: Update multiple and multiple_async methods to include agent configuration for improved task handling (`5ac116c6`)
- feat: Include Task Agent name in task analysis and decomposition prompts for improved context (`c40529f2`)
- docs: Update README.md to reflect changes in Direct and Task usage for LLM calls (`a25ecc81`)
- feat: Refactor reliability processing to enhance async validation execution and streamline context handling (`fd47ccab`)
- feat: Add tool validation and default tool propagation in agent configuration (`cb47f5a9`)
- feat: Add parallel task execution methods to agent configuration (`2be37f0e`)
- feat: Extend async support with thread-safe coroutine execution across client modules (`6be1bcce`)
- feat: Add comprehensive async support across client modules (`a384d5ff`)
- feat: Enhance server management and tool execution with robust server lifecycle handling (`ac214ec2`)
- chore: Upgrade mcp package to version 1.3.0 (`1dfbee14`)
- fix: Simplify tool registration condition in agent configuration (`dee840a3`)
- Update README.md (`cc6db853`)
- chore: Update markitdown package to version 0.0.1 (`c9f1b7f0`)

## 0.47.5
Released on 2025-03-07.

- bump: Increment package version to 0.47.5 (`df99e080`)
- feat: Add async reliability processing for reliability processes (`f728f0f2`)

## 0.47.4
Released on 2025-03-06.

- bump: Increment package version to 0.47.4 (`3b125282`)
- refactor: Optimize context initialization in Call and Agent classes (`68df34e8`)

## 0.47.3
Released on 2025-03-06.

- bump: Increment package version to 0.47.3 (`30f8f002`)
- fix: Improve KnowledgeBase RAG handling and context processing (`cc6e17a6`)

## 0.47.2
Released on 2025-03-05.

- bump: Increment package version to 0.47.2 (`4c1e2125`)
- feat: Add Crawl4AISimpleCrawling tool for web content extraction (`100537ee`)

## 0.47.1
Released on 2025-03-05.

- bump: Increment package version to 0.47.1 (`d54d412b`)
- feat: Add task duration tracking with start and end time properties (`74571acc`)

## 0.47.0
Released on 2025-03-05.

- bump: Increment package version to 0.47.0 (`59d27b22`)
- feat: Add total_cost property to Task for retrieving estimated task cost (`97681b3b`)
- feat: Add RAG capabilities to KnowledgeBase with LightRAG integration (`c45d87b4`)

## 0.46.1
Released on 2025-03-04.

- bump: Increment package version to 0.46.1 (`8878b903`)
- feat: Improve sub-task agent configuration and memory handling (`e93f5dab`)

## 0.46.0
Released on 2025-03-04.

- bump: Increment package version to 0.46.0 (`626e8320`)
- Update README.md (`f0ef8a9a`)
- feat: Add YoutubeSearch tool for retrieving YouTube video search results (`0bf3ceb5`)
- feat: Add YouTubeVideo tool for extracting video metadata and captions (`0d98a012`)
- feat: Add ArxivTool for academic paper search and retrieval (`d60b0d65`)
- feat: Enhance dependency and API key management for search and web tools (`6a898e2d`)
- feat: Add YFinanceTool for stock market data retrieval (`c7ef476e`)
- feat: Add Firecrawl web search, scraping, and crawling tools (`465c805a`)
- feat: Add SerperDev search utility class to tools module (`cfd8d4f4`)
- feat: Add debug print statements for tool call process (`11c32feb`)
- fix: Add traceback printing for better error debugging in tool calls (`ba046d9a`)
- fix: Improve tool validation to handle instance and class-based tools (`5da46ed7`)
- feat: Add DuckDuckGo search utility class to tools module (`69eb8d04`)
- feat: Add tool validation method to Task class (`42b6d005`)
- feat: Add Wikipedia utility class for search and summary operations (`b132f7a9`)
- Update README.md (`588addc7`)

## 0.45.4
Released on 2025-02-28.

- chore: Bump package version to 0.45.4 (`41cd4127`)
- feat: Pass images to validator and editor tasks in reliability processor (`abe773fe`)

## 0.45.3
Released on 2025-02-27.

- chore: Bump package version to 0.45.3 (`8021da64`)
- refactor: Restructure Direct LLM call class with static and instance-based approaches (`0759e608`)
- feat: Add Azure GPT-4o Mini support across pricing and model configurations (`b74d156e`)
- doc: Some updates for why (`2d935bdf`)

## 0.45.2
Released on 2025-02-26.

- chore: Bump package version to 0.45.2 (`d985bdd9`)
- doc: Fixed typos (`1e90858d`)
- feat: Add system prompt configuration for agent characterization (`df025ba4`)

## 0.45.1
Released on 2025-02-25.

- chore: Bump package version to 0.45.1 (`1c5f8296`)
- refactor: Improve dev server startup with parallel threading and robust error handling (`054f3647`)
- refactor: Enhance port availability and server startup performance (`3e26f85c`)
- refactor: Optimize server process management and port cleanup logic (`33e2b678`)
- refactor: Remove Sentry SDK tracing from bulk configuration setting method (`dc204d8f`)
- feat: Implement bulk configuration setting for client and server (`c5970fea`)
- fix: Correct server connection timing log placement in client initialization (`588a0a6d`)

## 0.45.0
Released on 2025-02-25.

- refactor: Optimize MarkItDown import and initialization in server modules (`05dbad8a`)
- feat: Add server connection timing measurement to initialization process (`4b17ec09`)
- Update README.md (`f7c987b4`)
- Update README.md (`aefdc3d0`)
- refactor: Improve exception handling in Call and Agent classes (`c9067ad2`)
- chore: Bump version (`5fff4a5d`)
- refactor: Restructure URL validation and context processing in reliability processor (`37604045`)
- refactor: Enhance URL validation logic with explicit handling of empty URL scenarios (`ac20d2f2`)
- refactor: Improve validation prompt handling for edge cases with no source data (`e7ed3b16`)
- feat: Add image support to task processing across client and server components (`63b74ff4`)
- feat: Add debug parameter to print_do method in direct LLM call (`23eedba5`)
- feat: Add price tracking and summary functionality for agent tasks (`ea2c0ff1`)

## 0.44.2
Released on 2025-02-19.

- chore: Bump package version to 0.44.2 (`1cec7627`)
- Update README.md (`58ea37e9`)
- refactor: Remove debug print statement in reliability processor context handling (`da93be3e`)
- refactor: Optimize task configuration and context handling in multi-task agent processing (`b584eb14`)
- refactor: Disable context compression in agent configuration (`3d58d526`)
- refactor: Clarify AI response context labeling in validation tasks (`06006849`)
- refactor: Enhance trusted source guidelines across verification tasks (`d47b583a`)
- refactor: Disable reflection and adjust agent configuration parameters (`a71da665`)
- refactor: Improve context processing for validation tasks in reliability processor (`adb1845c`)
- refactor: Simplify validation guidelines for numbers, code, and information verification (`bb878995`)
- refactor: Simplify URL source verification guidelines in reliability processor (`f4ffc3e9`)
- refactor: Refine source and number verification guidelines in reliability processor (`62ff1332`)
- refactor: Improve number verification guidelines in reliability processor (`05975d8d`)
- refactor: Streamline editor task prompt for more precise content validation (`c400e3a4`)
- feat: Enhance reliability processor with comprehensive validation points and improved suspicion detection (`7985c9ab`)
- feat: Refactor reliability processing with enhanced validation and verification mechanisms (`2edf64bd`)
- feat: Expand validator task prompt with comprehensive validation checks for URLs, identifiers, and data types (`2e1f37db`)
- Update README.md (`8a73bc58`)
- Update README.md (`1ce7e366`)
- Update README.md (`b85fe6bc`)

## 0.44.1
Released on 2025-02-15.

- chore: Bump package version to 0.44.1 (`b51de4c7`)
- feat: Enhance reliability processing with detailed validation and editing prompts (`de5bf55e`)
- doc: Fixes llm_model parameter with model (`e72ca439`)
- doc: added reliability layer (`6ffd904a`)

## 0.44.0
Released on 2025-02-13.

- chore: Bump package version to 0.44.0 (`41668d84`)
- feat: Improve context handling and validation in agent creation and reliability processing (`1d6e30c0`)
- feat: Propagate tools to validation and editing tasks in reliability processing (`84ef487e`)
- feat: Implement reliability processing for agent task results (`3681c086`)
- feat: Enhance agent configuration and server logging with debug options (`6809230f`)
- feat: Initialize Upsonic tools module with placeholder classes (`ed546283`)
- refactor: Replace pickledb with SQLite for configuration and caching (`489fe511`)

## 0.43.0
Released on 2025-02-09.

- chore: Bump package version to 0.43.0 (`185da039`)
- fix: Correct Azure OpenAI API key configuration variable (`ae124596`)
- refactor: Remove unused reasoning attribute from AgentMode class (`f963cbcb`)
- feat: Enhance BrowserUse agent with expected output parameter (`729c8fb9`)
- feat: Update LLM configuration with improved Azure OpenAI and model mappings (`4e94f775`)
- fix: Modify BrowserManager to return browser instance directly (`32724fb9`)
- chore: Reorder Dockerfile package installation steps (`630276e9`)
- feat: Disable GIF generation in BrowserUse agent (`ffc92705`)
- chore: Update Playwright installation command in Dockerfile (`ec14ba35`)
- chore: Remove explicit version constraints for LangChain dependencies (`33700a50`)
- chore: Update LangChain package dependencies to latest versions (`8e4433e9`)
- feat: Add BrowserUse tool with Playwright integration and Docker configuration (`b7aa68b9`)
- feat: Enhance ComputerTool with advanced display scaling and resolution management (`156eba66`)

## 0.42.0
Released on 2025-02-08.

- chore: Bump package version to 0.42.0 (`3ec77551`)
- feat: Improve multi-agent task execution with custom client support (`326e6709`)
- chore: Bump package version to 0.41.2 (`841e44d1`)
- feat: Add latest tag support to Docker build and publish workflow (`81d1f633`)

## 0.41.1
Released on 2025-02-08.

- chore: Bump package version to 0.41.1 (`79250968`)
- feat: Add multi-architecture Docker build and publish workflow (`af59f9a6`)
- fix: Refine Claude model version string matching in agent configuration (`631efc4d`)
- refactor: Simplify ComputerTool error handling and return structure (`85132144`)
- feat: Add model-specific settings for OpenAI and Anthropic models (`7f9d3aa8`)
- fix: Refine Claude model version string matching logic (`a36c00fd`)
- fix: Improve Claude model version string matching (`83ec1c88`)
- feat: Enhance ComputerTool with detailed action result feedback (`db5d3cce`)
- fix: Improve Docker manifest push reliability with authentication and retry mechanism (`4f976504`)
- refactor: Optimize Docker build process for multi-arch test server images (`31b81921`)
- refactor: Enhance Docker manifest creation with explicit architecture annotations (`59fe009e`)
- feat: Add multi-architecture Docker manifest creation for test server images (`b343f80f`)
- Update README.md (`e40dbe39`)
- fix: Update ARM build runner to ubuntu-24.04-arm (`37714175`)
- fix: Remove redundant directory copy steps in test_publisher workflow (`70045988`)

## 0.41.0
Released on 2025-02-07.

- fix: Typos (`6e735da3`)
- fix: Conditionally apply model settings only when tools are present (`bab9dd1d`)
- refactor: Simplify model settings configuration for GPT-4o and related models (`755439a1`)
- bump: Increment package version to 0.41.0 (`224a0020`)
- refactor: Improve timeout handling with signal-based mechanism across servers (`8f3e1b48`)
- feat: Add support for GPT-4o Mini model in pricing and agent creation (`058dddb7`)
- refactor: Migrate to fully async AI call and agent management (`4ace650f`)
- feat: Add graceful signal handling for ConfigManager database (`d923d28f`)
- Add files via upload (`9d2e6d43`)
- feat: Replace Google search with DuckDuckGo search tool (`c72c992d`)
- feat: Enhance tool registration and serialization for object instances (`8d010ce9`)

## 0.40.7
Released on 2025-02-04.

- bump: Increment package version to 0.40.7 (`5bb3daf8`)
- fix: Update port detection and process termination in ServerManager (`2c9bcc34`)

## 0.40.6
Released on 2025-02-03.

- bump: Increment package version to 0.40.6 (`99763647`)
- refactor: Remove redundant global client reference updates (`c133f26d`)
- fix: Ensure proper client initialization in get_or_create_client method (`8e7b41b2`)
- refactor: Improve port cleanup process using psutil for robust process termination (`77fe86fa`)
- fix: Enhance client status check in get_or_create_client method (`aa5b9bc4`)
- refactor: Rename llm_model parameter to model in Direct LLM call methods (`bb2ffdee`)
- refactor: Simplify latest_upsonic_client import and global management (`8ba76933`)
- feat: Improve port management and process cleanup in ServerManager (`2533f165`)
- docs: Add MultiAgent section to README with task distribution example (`58468e20`)
- Add MultiAgent support and update model settings for parallel tool calls (`a4ffb401`)
- docs: Added docs link (`9b2293f3`)
- removed old informations (`ac6671f8`)
- Update for dependencies (`e4188e59`)
- doc fixed the imports (`e4631e7f`)

## 0.40.5
Released on 2025-02-02.

- Bump version to 0.40.5 (`c56e2dc0`)
- Update dotenv loading to use current working directory (`f4780ac7`)

## 0.40.4
Released on 2025-02-02.

- Bump version to 0.40.4 (`dd039b73`)
- Enhance task decomposition with intelligent mode selection and tool-aware subtask generation (`42648527`)
- Enable reflection by default in agent configuration (`4d213e00`)

## 0.40.3
Released on 2025-02-02.

- Bump version to 0.40.3 (`c8374928`)
- Refactor context handling in agent and utility modules (`dabcfb39`)

## 0.40.2
Released on 2025-02-01.

- Bump version to 0.40.2 (`69ac1018`)

## 0.40.1
Released on 2025-02-01.

- Remove debug print statements from subtask context sharing (`17b6fa5c`)
- Improve subtask generation prompt and add focus directive for task completion (`75ba1817`)
- Bump version to 0.40.1 (`ae525b40`)
- Remove debug print statements from tool registration process (`da030def`)
- Enhance tool registration with improved type and module detection (`cedaea6e`)

## 0.40.0
Released on 2025-02-01.

- Bump version to 0.40.0 (`d398ab59`)
- Update README with simplified agent usage and configuration examples (`d08f4f2e`)
- Add Direct LLM call functionality with dynamic tool registration (`1a0a566c`)
- Add tool registration logic to AgentConfiguration with dynamic method detection (`72c6e5d9`)
- Add flexible initialization for AgentConfiguration and Task classes (`cb0cf134`)
- Add latest_upsonic_client global variable for dynamic client access (`9fe16309`)
- Enhance agent characterization with optional configuration parameters (`6ae0d8b1`)
- Return task/agent response instead of boolean in Call and Agent classes (`3e33d2b7`)
- Enhance UpsonicClient initialization with flexible configuration and improved server handling (`1c002276`)
- Add default model configuration to AgentConfiguration and Agent (`72fc6906`)
- Add flexible run method to UpsonicClient for simplified agent and call interactions (`ec883831`)
- Add Agent alias for AgentConfiguration in __init__.py (`d4de60b6`)

## 0.39.0
Released on 2025-02-01.

- New version for o3-mini (`4aaa0499`)

## 0.38.1
Released on 2025-01-31.

- Optimize GitHub Actions workflow to trigger tests only on relevant file changes (`0d9416b0`)
- New version (`185e7005`)
- Fix deepseek model condition in agent creator (`e61e9c51`)
- doc knowledge base example fixed (`5a5b88e9`)
- doc added deepseek information to other LLM's (`40d33bef`)
- Update README.md (`f3ee6201`)

## 0.38.0
Released on 2025-01-25.

- Bump version to 0.38.0 (`15404a58`)
- Remove debug print statement in config method of Storage class (`8480fcb4`)
- Centralize exception handling by creating a dedicated exception module (`4ef0d406`)
- Improve error message clarity for unexpected model behavior (`66c3ea05`)
- Improve context serialization with datetime and robust handling (`5aa53754`)
- Add ClientConfig for managing LLM and cloud service configurations (`641917fd`)
- Pass LLM model to GPT-4o call in agent satisfaction check (`4bd818d8`)
- Update pricing for deepseek-chat model to reflect new input and output costs (`06542d08`)
- changed 'deepseek-reasoner' to 'deepseek-chat' (`d7f79437`)
- Integrated deepseek-reasoner (`ddd4a28c`)
- Update README.md (`1db54ed7`)
- Enhance task context handling in Agent class by appending task context to subtasks. This change improves the management of task contexts, ensuring that subtasks have access to the relevant context information. (`a0a37113`)

## 0.37.0
Released on 2025-01-23.

- Update project version to 0.37.0 in pyproject.toml (`2d9745f6`)
- Refactor printing logic to enhance output for various result types, including search results, company objectives, and human objectives. Introduced a new characterization_print function for structured display of human and company information. Updated agent configuration to disable caching. Improved agent's create_characterization method to utilize the new printing function. (`39a6bea3`)
- Enhance agent functionality by adding subtask handling and retry mechanism. Updated printing logic to display subtasks and their details. Modified agent configuration to adjust reflection return value. Improved error handling in agent request process with retry capability. (`5feb4ffc`)
- doc added basic example and hard example to readme (`c34193ca`)
- doc new version added (`559bc604`)

## 0.36.0
Released on 2025-01-22.

- doc new version added (`d42cc5d8`)
- Enhance MCP tool management by adding error handling and logging. (`2f89f376`)
- doc new version added (`a0d81064`)
- Enhance error handling in call_agent function to manage unexpected model behavior. (`89d5421d`)
- Remove redundant result_retries parameter from run_sync_agent function in server.py (`7eea7bd1`)
- Fix (`38834f34`)
- doc new version added (`1823af0e`)
- Fix type hint for response_format in Task model to allow None as a valid type (`2c87ac4f`)
- doc new version added (`ea6ef88e`)
- Update parallel_tool_calls logic in CustomOpenAIAgentModel to handle absence of tools gracefully (`2adb76ec`)
- doc new version added (`0781c544`)
- Fix status codes in error responses for message compression and agent request processing (`23fa742f`)
- doc fixed installation (`440bcd89`)
- Context compress deafulty activated (`4c885284`)
- increased width of prints (`795666a0`)
- doc new version added (`58b529f6`)
- changed model defination (`fb89a8bd`)
- doc new version added (`27ce7911`)
- Enhance OpenAI model integration by introducing CustomOpenAIModel. (`6d24c576`)
- Fixed dependency version (`5ae04030`)
- Update pydantic-ai and pydantic-ai-slim to version 0.0.19; add pydantic-graph package with dependencies (`1ee92cc2`)
- Refactor error handling in AgentManager to specify exceptions and improve response messages. (`3e1b2319`)
- doc new version added (`77e9f30e`)
- Fix (`e5e92fa1`)
- doc new version added (`eb3a61f4`)
- Update agent prompt to clarify user request handling and avoid assumptions (`5a8acb9e`)
- Refactor client modules to improve debugging and output formatting (`75c697b5`)
- Enhance agent creation and context summarization functionality (`2dfb89a2`)
- fix typo (`ba56acd3`)
- Update README.md (`b4dde249`)
- doc updated readme example (`f37e4d0c`)
- Update README.md (`1e088095`)
- Update README.md (`e9c3ddf4`)
- Refactor Sentry spans in client modules: Remove span names for serialization, preparation, sending, and deserialization in Call, Agent, and Storage classes for consistency and clarity. (`9f5235f3`)
- Update pyproject.toml: Bump version to 0.36.0, enhance project description, and consolidate dependencies into the main section. (`71ea0a7b`)
- Enhance telemetry configuration in README.md and trace.py (`5439534e`)
- Refactor trace.py: Use environment variable for Sentry DSN configuration (`5e2f2f01`)
- Fix added latest version (`6f9010b6`)
- Refactor README.md: Update class names to follow naming conventions, enhance task1 output with detailed blog suggestions, and clarify tools list with HackerNewsMCP reference. (`2e5769e1`)
- Fixed the size of other llms text (`f2d03022`)
- Update README.md to include configuration examples for additional LLMs and change client server to 'localserver' (`dba9a765`)
- Refactor server type detection to include 'localserver' in UpsonicClient (`838401d2`)
- Update README.md (`90f32829`)
- Update README.md (`58810299`)
- Update README.md (`aa8a3e9f`)
- Update README.md (`e486fb1e`)
- fix (`a00d206c`)
- Update README.md (`07573b2b`)
- Fix (`aad9b9ae`)
- Update README.md (`6a8c0304`)
- Update README.md (`159c74e7`)
- Fİx (`f3cd4a27`)
- Fix (`5e8ca488`)
- Fix (`887ada83`)
- Update README.md (`121c2d72`)
- Update README.md (`f2265e65`)
- Update README.md (`43b3315f`)
- Update README.md (`3099ccea`)
- Fix (`1deee347`)
- Update README.md (`e1d9983e`)
- Update README.md (`7956e674`)
- Fix (`cc1377ae`)
- Update README.md (`63374710`)
- Update README.md (`15c331fc`)
- New (#247) (`12cde7c2`)
- Update README.md (`3ef69545`)
- Update README.md (`89773f4f`)
- Update README.md (`680436ad`)
- Update README.md (`a3cb5ba3`)
- Update README.md (`d1729b2c`)
- Turning into Upsonic (#245) (`8c287304`)
- Update README.md (`9c4c37e5`)
- Update README.md (`edaff060`)
- Update README.md (`74e11948`)
- Update README.md (`ed76cbb0`)
- Update README.md (`c885ffb2`)
- Create test_publisher.yml (`7f81fccf`)

## 0.28.3
Released on 2024-12-31.

- Changed version number with v0.28.3 (`11f39aab`)
- feat: Refactor cloud_instance to streamline user ID handling and clean up tool list in agent (`c7a9df77`)

## 0.28.2
Released on 2024-12-31.

- Changed version number with v0.28.2 (`010bcf26`)
- feat: Add Sentry SDK for improved error tracking and monitoring (`1265a68c`)

## 0.28.1
Released on 2024-12-31.

- Changed version number with v0.28.1 (`c6fcbde4`)
- feat: Integrate user ID handling in cloud instance and Sentry for enhanced tracking (`f2017f21`)

## 0.28.0
Released on 2024-12-31.

- Changed version number with v0.28.0 (`2499d963`)
- feat: Add user ID management functionality with API endpoints (`eb374094`)

## 0.27.10
Released on 2024-12-31.

- Changed version number with v0.27.10 (`afb9714d`)
- fix: Increase recursion limit to 100 and max retries to 35 for enhanced performance (`cbe050e4`)

## 0.27.9
Released on 2024-12-31.

- Changed version number with v0.27.9 (`fcb18a28`)
- refactor: Replace chat_agent_executor with create_react_agent for improved clarity (`21a98ec5`)

## 0.27.8
Released on 2024-12-31.

- Changed version number with v0.27.8 (`a4ee4a21`)
- fix: Clean up version string parsing to remove formatting artifacts (`dd22bd90`)

## 0.27.7
Released on 2024-12-31.

- Changed version number with v0.27.7 (`4dc7a1b7`)
- fix: Increase recursion limit from 10 to 40 for improved performance (`6a73a8f8`)

## 0.27.6
Released on 2024-12-30.

- Changed version number with v0.27.6 (`eda13644`)
- feat: Integrate Sentry SDK for error tracking and performance monitoring (`c562d0cd`)
- fix: Remove unnecessary print statements related to sentry-sdk initialization (`0a9a1222`)

## 0.27.5
Released on 2024-12-30.

- Changed version number with v0.27.5 (`c4862662`)
- fix: Ensure artifacts directory is created if it does not exist (`5fa5fdcc`)

## 0.27.4
Released on 2024-12-30.

- Changed version number with v0.27.4 (`aad5ebc5`)
- fix: Update sentry-sdk dependency format in requirements.txt (`5b3e5a97`)

## 0.27.3
Released on 2024-12-30.

- Changed version number with v0.27.3 (`fd88143e`)

## 0.27.2
Released on 2024-12-30.

- Changed version number with v0.27.2 (`3782e22b`)
- fix: Correct sentry-sdk dependency format and update install_requires in setup.py (`81beef95`)

## 0.27.1
Released on 2024-12-30.

- Changed version number with v0.27.1 (`136f027d`)
- refactor: Rename instance methods to agent for consistency across modules (`06298b0a`)
- refactor: Remove test function for Sentry transaction from classes.py (`2f4a3381`)

## 0.27.0
Released on 2024-12-30.

- Changed version number with v0.27.0 (`57ce34ed`)
- feat: Add release version to Sentry SDK initialization for better tracking (`a0f5d95e`)
- feat: Introduce tracing module and update requirements for Sentry SDK integration (`8db2c7a4`)
- fix: Clean up output formatting in Task class for verification messages (`4ac9e0b1`)
- feat: Add method to send system messages to cloud instance (`386f875a`)
- fix: Simplify retry attempt message in Task class for cleaner output (`751154a5`)
- fix: Remove traceback printing on verification failure for cleaner error handling (`8dac9819`)
- feat: Update README.md to include Task and TypeVerifier imports for clarity (`c850edb3`)
- feat: Update README.md to include task addition and execution for GitHub star count (`4d87cb4c`)
- fix: Capture AI output in verification error messages for better debugging (`c3137d70`)
- feat: Add profile change and log retrieval methods; enhance console output with rich styling (`0d25c106`)
- Replacing save_models with save_model (#240) (`d4ff141c`)
- fix: Updated Contributors profile link in README.md (`71b5988d`)

## 0.26.11
Released on 2024-12-21.

- Changed version number with v0.26.11 (`73ef0914`)
- fix: Update mcp_tools function to initialize the_tools_ as an empty list (`babed445`)
- feat: Implement task management and SHA-256 hashing in verifier (`1c1c288e`)

## 0.26.10
Released on 2024-12-21.

- Changed version number with v0.26.10 (`c8f5ba2a`)
- feat: Add no_tools parameter to get_agent_executor and update screenshot_ function to use it (`3ff80378`)

## 0.26.9
Released on 2024-12-20.

- Changed version number with v0.26.9 (`b2d54eb9`)
- fix: Reduce recursion_limit from 1000 to 10 for improved performance (`733c1601`)

## 0.26.8
Released on 2024-12-20.

- Changed version number with v0.26.8 (`ef550340`)
- feat: Add run_terminal_command function to execute terminal commands (`91cf0849`)
- feat: Enhance instance classes with task management and add reset_memory method (`d7f79977`)

## 0.26.7
Released on 2024-12-19.

- Changed version number with v0.26.7 (`ecb30479`)
- feat: Add recursion_limit parameter to configuration (`e2a6e809`)

## 0.26.6
Released on 2024-12-18.

- Changed version number with v0.26.6 (`215f4c59`)
- fix: Add region_name parameter for AWS configuration (`3bd6c682`)

## 0.26.5
Released on 2024-12-18.

- Changed version number with v0.26.5 (`e72f6c5e`)
- feat: Add print statement to display the model being used in assistant function (`fc4877c0`)

## 0.26.4
Released on 2024-12-18.

- Changed version number with v0.26.4 (`a562f2cd`)
- fix: Remove max_retries parameter and update model identifier for AWS provider (`febc179e`)

## 0.26.3
Released on 2024-12-18.

- Changed version number with v0.26.3 (`c387dab5`)
- fix: Update condition for model selection based on AWS access key (`7f81286c`)

## 0.26.2
Released on 2024-12-18.

- Changed version number with v0.26.2 (`2bda45dd`)
- fix: Update model selection logic for AWS provider in agent executor (`30be32e0`)

## 0.26.1
Released on 2024-12-18.

- Changed version number with v0.26.1 (`7a666310`)
- fix: Update AWS credentials loading in ChatBedrock model configuration (`2e6e128e`)

## 0.26.0
Released on 2024-12-18.

- Changed version number with v0.26.0 (`98ab46a4`)
- feat: Add AWS provider support and related API endpoints (`3de930ba`)
- feat: Add Jupyter notebook for MCP integration with various tools (`9fe9bfdb`)
- feat: Add Jupyter notebook for MCP integration with various tools (`0ced2b50`)

## 0.25.2
Released on 2024-12-14.

- Changed version number with v0.25.2 (`7c2a6e47`)
- feat: Enhance click functions to support double-click functionality (`af88c4d8`)

## 0.25.1
Released on 2024-12-14.

- Changed version number with v0.25.1 (`5d35992a`)
- fix: Update SyncInvocationManager path to root for correct file system access (`c0352f52`)

## 0.25.0
Released on 2024-12-14.

- Changed version number with v0.25.0 (`563b42b3`)
- feat: Enhance request handling with polling for result retrieval (`f0d452ce`)
- Update README.md (`d7187ffc`)
- doc: Changed image (`e7328640`)
- docs: Improve README formatting by removing extra space in container startup section (`e33c6868`)
- docs: Update README formatting and improve clarity in container startup instructions (`f33708a5`)
- feat: Add Docker support and update README with usage instructions (`ffa609b8`)
- feat: Add functionality to manage custom MCP servers via API (`36ad8eac`)

## 0.24.33
Released on 2024-12-10.

- Changed version number with v0.24.33 (`8ed64a6b`)
- feat: Initialize chat with system prompt if chat history is empty (`39ad15d6`)

## 0.24.32
Released on 2024-12-10.

- Changed version number with v0.24.32 (`d6d417a5`)

## 0.24.31
Released on 2024-12-10.

- Changed version number with v0.24.31 (`1f8213fd`)
- feat: Add API endpoint to save system prompt and load it for initialization (`d9deb9d3`)
- feat: Add screenshot tool for capturing and analyzing images (`72f40588`)

## 0.24.30
Released on 2024-12-09.

- Changed version number with v0.24.30 (`3285a654`)
- feat: Add mouse_scroll tool and remove smooth_move_to function for streamlined interaction (`2fea742c`)

## 0.24.29
Released on 2024-12-09.

- Changed version number with v0.24.29 (`6442798c`)
- feat: Add keyboard_write and keyboard_press tools for enhanced keyboard interaction (`b16a0689`)

## 0.24.28
Released on 2024-12-09.

- Changed version number with v0.24.28 (`b274a5cf`)
- feat: Add click_to_area tool and enhance click_to_text and click_to_icon functionality (`0787ca19`)

## 0.24.27
Released on 2024-12-09.

- Changed version number with v0.24.27 (`8ecd3631`)
- feat: Introduce click_to_text and click_to_icon tools for enhanced interaction (`44d8c3c4`)

## 0.24.26
Released on 2024-12-09.

- Changed version number with v0.24.26 (`08ba4ada`)
- feat: Add read_website tool to fetch and parse website content (`f3219ada`)

## 0.24.25
Released on 2024-12-09.

- Changed version number with v0.24.25 (`e48c1b0d`)
- feat: Refactor computer tool actions and enhance documentation for clarity (`a3349ae7`)
- feat: Update interaction guidelines and enhance memory management in LLM settings (`91b96ccd`)
- feat: Integrate standard tools into agent executor and refactor MCP tools (`f5afd253`)

## 0.24.24
Released on 2024-12-09.

- Changed version number with v0.24.24 (`c267c6e3`)
- feat: Implement Anthropic API key management and enhance agent executor functionality (`0d9624bf`)
- feat: Enhance tool execution logging and error handling in MCPToolWrapper (`f2ae37c6`)
- feat: Add print statement to display result in MCPToolWrapper (`f0f6398f`)

## 0.24.23
Released on 2024-12-09.

- Changed version number with v0.24.23 (`95ebf1fa`)
- fix: Update import handling in stop_server function and change command in SyncInvocationManager (`889424f4`)
- feat: Add logging for tool execution in MCPToolWrapper (`439b2c29`)

## 0.24.22
Released on 2024-12-09.

- Changed version number with v0.24.22 (`eb79bd98`)
- fix: Correct screenshot path argument in process_text_api function call (`6f015bda`)

## 0.24.21
Released on 2024-12-09.

- Changed version number with v0.24.21 (`9bb697d8`)
- refactor: Remove redundant updates to the main window in profile and settings functions (`4359e56e`)

## 0.24.20
Released on 2024-12-09.

- Changed version number with v0.24.20 (`433c9173`)
- feat: Implement process_text_api function and integrate with the input handling in the API (`76d0e0d4`)

## 0.24.19
Released on 2024-12-09.

- Changed version number with v0.24.19 (`7d9681bd`)
- refactor: Simplify _run method by removing redundant import handling and animation logic (`2d145205`)

## 0.24.18
Released on 2024-12-08.

- Changed version number with v0.24.18 (`5e75430a`)
- feat: Add MCP tools integration and update requirements (`944abfdf`)
- doc: Fixed author and author email (`b3b3c241`)
- fix: Resolved errors on cloud demo script (`6dd9b745`)

## 0.24.17
Released on 2024-12-08.

- Changed version number with v0.24.17 (`6b0c5438`)
- fix: Correct typo in return statement in cloud class (`67715ce5`)

## 0.24.16
Released on 2024-12-07.

- Changed version number with v0.24.16 (`eaa35b72`)
- fix: Remove unused each_message_extension import and update related logic in agent and chat history (`4843df43`)

## 0.24.15
Released on 2024-12-07.

- Changed version number with v0.24.15 (`0429d79c`)
- fix: Remove unused import statement in assistant.py (`0da4b7f3`)

## 0.24.14
Released on 2024-12-07.

- Changed version number with v0.24.14 (`58afd7a5`)
- fix: Add random character generation to default response in assistant function (`f576348d`)

## 0.24.13
Released on 2024-12-07.

- Changed version number with v0.24.13 (`edd125ef`)
- fix: Handle empty content in ChatHistory by providing a default response (`baf84bab`)

## 0.24.12
Released on 2024-12-07.

- Changed version number with v0.24.12 (`6df040e5`)
- fix: Add debug print statements for assistant and system message types in ChatHistory class (`abe23396`)

## 0.24.11
Released on 2024-12-07.

- Changed version number with v0.24.11 (`6558f33a`)
- fix: Improve import handling and update KOT initialization in ChatHistory class (`746bcc1e`)

## 0.24.10
Released on 2024-12-07.

- Changed version number with v0.24.10 (`8c919c17`)
- fix: Reduce auto-delete duration for chat messages from 50 to 10 seconds (`577c4600`)
- feat: Implement ChatHistory class for managing chat messages and history (`3624c5c8`)

## 0.24.9
Released on 2024-12-07.

- Changed version number with v0.24.9 (`02cc3015`)
- refactor: Enhance error handling in computer_tool_ function with try-except block (`ea8c7956`)

## 0.24.8
Released on 2024-12-07.

- Changed version number with v0.24.8 (`92104377`)
- refactor: Simplify message invocation by removing unnecessary try-except block (`7d8e0baa`)

## 0.24.7
Released on 2024-12-07.

- Changed version number with v0.24.7 (`4a02a136`)
- fix: Handle None and empty content in message history with "No response" (`15af24fb`)

## 0.24.6
Released on 2024-12-07.

- Changed version number with v0.24.6 (`e1875f7f`)
- docs: Add instruction to avoid blank responses in each_message_extension function (`05da021a`)

## 0.24.5
Released on 2024-12-07.

- Changed version number with v0.24.5 (`97e7a1de`)
- fix: Replace empty message content with "No response" for Anthropic provider (`9e562e67`)

## 0.24.4
Released on 2024-12-07.

- Changed version number with v0.24.4 (`a33ab4dc`)
- feat: Add mouse_move_and_left_click action to computer_tool_ function (`3ff32547`)

## 0.24.3
Released on 2024-12-07.

- Changed version number with v0.24.3 (`01ce20b1`)
- fix: Set default value for last_message when empty (`2b464394`)

## 0.24.2
Released on 2024-12-07.

- Changed version number with v0.24.2 (`d71de019`)
- fix: Handle list return values in assistant function (`6920e521`)

## 0.24.1
Released on 2024-12-07.

- Changed version number with v0.24.1 (`04832cff`)
- feat: Add gpt_computer_assistant.cu package to setup.py (`967978e3`)

## 0.24.0
Released on 2024-12-07.

- Changed version number with v0.24.0 (`3469f57c`)
- Anthropic Computer Use (#232) (`f5630a2e`)
- Some styling (`3dd25493`)

## 0.23.27
Released on 2024-12-06.

- Changed version number with v0.23.27 (`a8ee6170`)
- feat: Enhance ocr_test_ and related functions with screen capture and binarization options (`cc7c13df`)
- Added screenshots and playground (`9b5d25e2`)

## 0.23.26
Released on 2024-12-06.

- Changed version number with v0.23.26 (`2523ac32`)
- feat: Update parameters for ocr_test_ and related functions for improved flexibility (`1207965d`)

## 0.23.25
Released on 2024-12-06.

- Changed version number with v0.23.25 (`c6c2f485`)
- feat: Add terminal_command tool to execute shell commands and return output (`ae5ecb9b`)

## 0.23.24
Released on 2024-12-05.

- Changed version number with v0.23.24 (`d97950d8`)
- feat: Add error handling to ocr_test_ function for improved robustness (`72120bcd`)

## 0.23.23
Released on 2024-12-05.

- Changed version number with v0.23.23 (`9469c3f9`)
- refactor: Remove unused text parameter from ocr_test_ function (`70ad6512`)

## 0.23.22
Released on 2024-12-05.

- Changed version number with v0.23.22 (`bef86c57`)
- feat: Add ocr_test tool for extracting text coordinates from the screen (`e8e362cd`)

## 0.23.21
Released on 2024-12-05.

- Changed version number with v0.23.21 (`735c1d5d`)
- feat: Add screen parameter to request methods in local_instance and cloud_instance classes (`bee44fe5`)
- feat: Add screen parameter to request handling in API and Remote_Client (`cca53295`)
- fix: Update pytesseract configuration for improved text extraction (`18110bfb`)

## 0.23.20
Released on 2024-12-05.

- Changed version number with v0.23.20 (`c3023f21`)
- chore: Remove opencv-python version from requirements files (`ba0c461f`)

## 0.23.19
Released on 2024-12-05.

- Changed version number with v0.23.19 (`b0986354`)
- chore: Pin opencv-python version to 4.10.0.84 in requirements files (`c787ca05`)

## 0.23.18
Released on 2024-12-05.

- Changed version number with v0.23.18 (`2cd8423d`)
- feat: Improve exception handling to provide detailed traceback in error messages (`e86dc366`)
- feat: Add exception handling to break loop on "EXCEPTION" response (`2e8acbbd`)
- chore: Update requirements to include opencv-python and screeninfo (`87fdb36b`)

## 0.23.17
Released on 2024-12-05.

- Changed version number with v0.23.17 (`5d72e0c7`)
- feat: Enhance request method to handle JSON parsing errors gracefully (`0969445d`)
- feat: Implement server stop functionality and enhance local_instance close method (`5ceab1ad`)
- feat: Add current_screenshot method to capture and display instance screenshots (`c87fd6c1`)
- chore: Add matplotlib as a dependency in setup.py (`431eb922`)
- feat: Enhance cloud_instance to manage instance lifecycle and suppress SSL warnings (`88b7fb40`)
- chore: Add requests package as a dependency in setup.py (`5c5a6e83`)
- chore: Update kot package version to 0.1.2 in requirements files (`27412e76`)
- New GCA in here. (#231) (`492aadf5`)

## 0.23.16
Released on 2024-12-03.

- Changed version number with v0.23.16 (`87f99f4d`)
- feat: Add extract_possible_coordinates_of_text function to identify text coordinates on screen (`26a1ba48`)

## 0.23.15
Released on 2024-12-02.

- Changed version number with v0.23.15 (`34becac7`)
- fix: Simplify get_client function by removing base_url parameter (`d01eb8e7`)

## 0.23.14
Released on 2024-12-02.

- Changed version number with v0.23.14 (`4c89b3c4`)
- fix: Remove base_url parameter from model configuration in get_model function (`fa511c54`)

## 0.23.13
Released on 2024-12-02.

- Changed version number with v0.23.13 (`0574cffb`)
- fix: Update import paths for the_input_box in record.py to ensure correct module resolution (`0cbf5f80`)

## 0.23.12
Released on 2024-12-02.

- Changed version number with v0.23.12 (`e25d3aeb`)
- fix: Change exception to print statement for server status check in Remote_Client (`7ec64360`)

## 0.23.11
Released on 2024-12-01.

- Changed version number with v0.23.11 (`bfcbd6de`)
- fix: Add try-except block for conditional imports in __init__.py (`30886568`)
- refactor: Move upsonic import inside the the_upsonic function for better encapsulation (`7e1901a6`)

## 0.23.10
Released on 2024-12-01.

- Changed version number with v0.23.10 (`4ebedebd`)
- refactor: Move PyQt5 imports inside the start function for better encapsulation (`eeebe2f1`)

## 0.23.9
Released on 2024-12-01.

- Changed version number with v0.23.9 (`9d683b71`)
- fix: Update data key in save_model_settings method to model_settings (`5314791c`)
- refactor: Move the_input_box_pre to a new module and update imports (`d4b06d79`)

## 0.23.8
Released on 2024-11-30.

- Changed version number with v0.23.8 (`ce4e4913`)
- fix: Add error handling to click sound initialization in the assistant (`0d335581`)

## 0.23.7
Released on 2024-11-30.

- Changed version number with v0.23.7 (`aa635a76`)
- fix: Change API server binding from localhost to 0.0.0.0 for external access (`cdb88233`)

## 0.23.6
Released on 2024-11-30.

- Changed version number with v0.23.6 (`e11c3f04`)
- feat: Update requirements to include opencv-python for display functionality (`de0d054b`)
- feat: Add just_screenshot option to assistant function for screenshot handling (`8ff1ed55`)
- fix: Improve input handling and response validation in the assistant function (`7bf527d5`)
- feat: Add mouse scroll up and down API endpoints (`e3973f3d`)
- fix: Handle empty response in assistant function (`8fb0906a`)
- fix: Resolved display tools and azureai releation (`7a0d9f71`)
- fix: Resolved get_azureai_models overwriting (`1f410ce9`)
- feat: Added remote api_version functions and apis (`539a35bf`)

## 0.23.5
Released on 2024-11-29.

- Changed version number with v0.23.5 (`28af4089`)
- fix: Resolved non-mic systems (`753f5046`)

## 0.23.4
Released on 2024-11-29.

- Changed version number with v0.23.4 (`08fa3865`)
- fix: Resolved kot version (`756eb586`)

## 0.23.3
Released on 2024-11-29.

- Changed version number with v0.23.3 (`3f0706c0`)
- fix: Resolved dependency (`3f5791ea`)
- fix: Removed 's' (`f9e516a1`)

## 0.23.2
Released on 2024-11-29.

- Changed version number with v0.23.2 (`2bcceb7d`)
- fix: Removed fixed version for pyqt5 (`06215e42`)

## 0.23.1
Released on 2024-11-29.

- Changed version number with v0.23.1 (`2efb42ed`)
- fix: Removed build exe and dmg (`dd175189`)

## 0.23.0
Released on 2024-11-29.

- Changed version number with v0.23.0 (`236833ad`)
- feat: Added azureai models api (`d4dcef74`)
- fix: Added compability with copy pasted texts (`9eabb2ab`)
- feat: Added shift enter to move new line (`0fee3936`)
- fix: Resolved line to right problem (`5644c3fa`)
- feat: Added api version for azure models (`5a2668e2`)
- feat: Added kot db (`52da7069`)
- feat: Added azureai (`59fd6dfb`)
- Update README.md (`80d47e77`)
- Update README.md (`68b83d4a`)
- Update LICENSE (`0567eef8`)
- refactor: Scheduled refactoring (`3f5bee4f`)
- fix: Fixed refactor.yml (`15f80d03`)
- feat: Added refactoring process (`2dacd111`)
- doc: Changed python version info with bold one (`4865fcdb`)
- doc: Added an information about python versions (`9af65ad2`)
- doc: Added producthunt social proof (`43cafe38`)

## 0.22.3
Released on 2024-08-14.

- Changed version number with v0.22.3 (`bda8c8c6`)
- fix: Hot fix for providers folder (`720ddeff`)

## 0.22.2
Released on 2024-08-14.

- Changed version number with v0.22.2 (`207dd6ea`)
- doc: Added producthunt link (`504a4d75`)
- chore: Add test_logo.png to .gitignore (`d043b321`)
- chore: Add test_backup.py to .gitignore (`07869c21`)
- doc: Added use case image (`49744d9a`)
- Update LICENSE (`44e3e679`)
- doc: Visual improvements (`d6700825`)
- doc: Removed some unused things (`1ad98593`)
- doc: Some structural changes (`0ef9676c`)
- doc: Update remote.input API documentation (`4c1fd5d9`)
- doc: Update README.md structure and content (`4f869a03`)
- doc: Changed readme structure (`225ef457`)
- doc: Update Native Applications status to Completed for Q3 2024 (`17d753e0`)
- feat: Enable TTS with bypass_other_settings flag (`61d2a08f`)
- doc: Added social media badges (`fa5d089d`)
- doc: Some structural changes (`87b3e146`)
- doc: Some structural changes (`3a84cada`)
- doc: Changed images (`1ca319ae`)
- Update README.md (`12ebbdd0`)
- feat: Add save_model method to Remote_Client class (`00dfafc0`)
- refactor: get_openai_models function to remove unnecessary character in function name (`c1d16dbc`)
- feat: Added get x provider models remote API's (`49e54527`)
- Refactor minimize and close buttons in title bar for better user experience (`ca3d81b4`)
- Refactor build scripts for macOS and Windows (`acbef881`)
- Refactor build scripts for macOS and Windows (`c217a420`)
- Refactor build scripts for macOS and Windows (`bb6f86f7`)
- chore: Update release_generation.yml workflow to include workflow_dispatch trigger and download DMG and EXE artifacts (`d559a1bb`)
- chore: Update release_generation.yml workflow to include workflow_dispatch trigger and download DMG and EXE artifacts (`1e7b44e2`)
- chore: Update release_generation.yml workflow to include workflow_dispatch trigger and download DMG and EXE artifacts (`8c83a3e6`)
- chore: Update release_generation.yml workflow to include workflow_dispatch trigger and download DMG and EXE artifacts (`7194f176`)
- chore: Download DMG and EXE artifacts in release_generation workflow (`11a1a3fd`)
- feat: Update release_generation.yml workflow to include workflow_dispatch trigger (`56d38917`)
- Refactor build scripts for macOS and Windows (`9a0b4ef3`)
- chore: Update release_generation.yml workflow to include workflow_dispatch trigger (`46c1552b`)
- Refactor build scripts for macOS and Windows (`a31fba86`)

## 0.22.1
Released on 2024-08-10.

- Changed version number with v0.22.1 (`d13643e5`)
- feat: Add build scripts for macOS and Windows (`d6217c70`)

## 0.22.0
Released on 2024-08-10.

- Changed version number with v0.22.0 (`a5c52dc2`)
- feat: Added pip3 install gcadev option (`133d7b3f`)
- feat: Refactor collapse and long GCA functionality (`5084810d`)
- feat: Improve UI layout and expand button functionality (`166f4eac`)
- feat: Added train remote api and bacground system (`65314c87`)
- feat: Added default_logo remote API (`e0071bb8`)
- feat: Add Python&Markdown syntax highlighting to input box (`55c718c1`)
- feat: Update input box styling and font size (`da0c5c2d`)
- feat: Added default right side center positioning (`0e3dbd79`)
- feat: Changed default tts situation on API, changed (`72cd151d`)
- feat: Added long gca mode (`e36e059b`)
- fix: Fixed buggy show and hide logo proccess (`9416246e`)
- feat: Added custom_logo_upload remote API (`9f371486`)
- feat: Added show_logo and hide_logo remote apis (`4125b123`)
- feat: Added an logo file path (`d5f32395`)
- feat: Added an logo place for custom logo setting (`6560b568`)
- chore: Specified the png files that shouldnt be sent (`cd0946b7`)
- fix: Exe creation (#218) (`0e5a8517`)
- chore: Update Windows build script to import PyQt5.sip for Windows pyinstaller compatibility (`380abd45`)
- chore: Import PyQt5.sip for Windows pyinstaller compatibility (`cc19b5c5`)
- chore: Import PyQt5.sip for Windows pyinstaller compatibility (`c1fca99c`)
- chore: Update Windows build script to use 'python -m pip' for package installations (`1a8f2a89`)
- feat: Add Windows build workflow for OpenAI EXE (`4b7b21d8`)
- feat: Add workflow_dispatch trigger for OpenAI DMG build (`1bb261e6`)
- feat: Added save_stt_model_settings remote API (`e573dd7b`)
- feat: Added save_tts_model_settings remote API (`116075d9`)
- feat: Added save_google_api_key remote API (`d4a114e1`)
- feat: Added save_groq_api_key remote API (`bc524d6c`)
- feat: Added save_model_settings remote API (`dd6f940d`)
- feat: Added save_openai_url remote API (`888c0494`)
- feat: Added openai api key setting remote API (`94b5e677`)
- doc: Added new API's informations (`9346dbdd`)
- feat: Small change for usage of set_background_color function of remote (`ed549cfb`)
- feat: Added collapse and uncollapse remote API (`70d22a16`)
- fix: Some improvements for title bar (`7fa552f8`)
- fix: Resolved top bar radius (`dafdc37e`)
- feat: Added set_border_radius remote API (`cc4965d2`)
- feat: Added set_opacity and set_background_color remote API's (`ef5d23cf`)
- Update README.md (`c7ef15e0`)
- revert: Solved the workflow file that been deleted (`d63e0fff`)
- chore: Removed old unused workflow trash files (`e8e233c4`)
- fix: Resolved location problem of build_openai_dmg workflow (`5fcd512f`)
- fix: Resolved api set text function (`b3b446d7`)
- perf: Improved remote load time via cached tiger (`5d74d2af`)
- feat: Added set_text function to remote (`8d1f5321`)
- feat: Added set_text api (`6b7cb830`)
- doc: Added an example for ask api (`3e28cba3`)
- feat: Added ask api (`aa0762a9`)
- fix: Added required wrappers for ask_to_user ability (`9acdf524`)
- feat: Added ask ability to llm (`93770d1a`)
- refactor: Some import settings for development envinronment (`bc281d57`)
- refactor: Generalized stop ai talking functionality (`84bb6365`)
- feat: Added boop sound API (`7210a0f2`)
- doc: Added information about remote.operation api (`7b366071`)
- feat: Added operation animation and apis (`4d148079`)
- feat: Changed gobal hotkey (`18864d49`)
- doc: Some roadmap changes (`23ad7861`)
- feat: Added global hotkey (`ee1c7dbc`)
- feat: Removed send and screenshot buttons (`603ffaeb`)
- chore: Added an powered by label (`af791adb`)
- feat: Aded agentic infra to macos openai build (`0239ca3c`)
- refactor: Redesigned build script locations (`25c5d03f`)
- fix: Resolved AppOpener confict with linux (`898b66eb`)
- fix: Added `pip install pyinstaller==6.9.0` to build dmg worflow (`b7ffd2fe`)
- fix: Added `brew install create-dmg` to build dmg workflow (`2039487e`)
- feat: Added menu bar (`05f3b355`)
- fix: Resolved name problem between workflow and script location (`14316c5e`)
- feat: Added  Build DMG workflow (`b92053ab`)
- feat: Aded .dmg file generation script for macos (`9ab48832`)
- feat: Added icon that compatible with macos on builds (`35c22945`)
- fix: Resolved icon showing in different os (`71b49434`)
- fix: Fixed radius of icons (`0296d923`)
- feat: Added ability to set tts, stt provider and llm model from cli (`6e0fe010`)
- doc: Added installation doc for local stt and tts (`2d93139d`)
- chore: Added .DS_Store to gitignore (`5ce1e410`)
- feat: Added wifi turn on, off and connect tools (`9f2a48f7`)
- feat: Added get_texts_on_the_screen tool (`080e8260`)
- fix: Resolved variously problems answer detection system (`f158e974`)
- feat: Added an prereload on local model support (`f52cf870`)
- feat: Added llama3.1 (`00a64cfe`)
- fix: Resolved local model tool calling problem (`f47af409`)
- doc: Changed local stt status (`4ac0a45a`)
- fix: Resolved first message problem (`fe8f83c6`)
- fix: Resolved vision situations on gui (`a411abef`)
- feat: Hide screenshot button when vision not available (`b12ccaae`)
- refactor: Removed old audio and vision system for gui (`a6b4fc03`)
- feat: Aded local local stt (`ada2ad8b`)
- doc: Changed local tts situation (`16ed7700`)
- feat: Added local tts (`b4605276`)
- refactor: Some code improvements on tts mechanism (`42764e26`)
- doc: Some new features (`327964a1`)
- refactor: Improved llm setting system (`c50ac0cd`)
- fix: Resolved phi3 1.5b to qwen2 1.5b (`aeea174b`)
- feat: Added an automation for llm setting system (`607c5295`)
- feat: Added support to phi3 1.5b (`6f6dbaf3`)
- Added `pip3 install setuptools --update` to README (`8e2c760d`)
- fix: Resolved screenshot api (`054dc3e4`)
- feat: Added an status control for remote (`526badc8`)
- doc: Added informations about latest install_library and custom_tool mechanism of rmeote (`75c41c3d`)
- feat: Added remote.custom_tool mechanism (`92dfa196`)
- feat: Enable custosm tool set dynamic (`f321006c`)
- feat: Added str_to_function function to utility.functions (`fc518966`)
- feat: Added install_library and uninstall_library API's (`bf530a62`)
- feat: Added change_name and change_developer API's (#210) (`f7ebe413`)

## 0.21.1
Released on 2024-07-23.

- Changed version number with v0.21.1 (`8f28a74a`)
- feat: Added wrappers to display tools (`67f8edb0`)
- fix: Added an time sleep to prevent tool name bugs (`e7e7586a`)
- feat: Added some details about assistant itself (`8bfba37d`)
- fix: Resolved duplication with openai (`71983bb3`)

## 0.21.0
Released on 2024-07-23.

- Changed version number with v0.21.0 (`bfdead98`)
- feat: Added gpt-4o-mini support (`f93727cd`)
- fix: Resolved ollama image system (`f219a408`)
- fix: Fixed ollama (`29ae58e9`)
- feat: Added llava-llama3 ollama support (`2570769e`)
- README.TR.md file created, Turkish README file provided. (#206) (`ab616f66`)
- feat: Added sound feedback for some operations (`79d45d41`)
- feat: Added an border animation to screenshot button (`6d680f63`)
- fix: Resolved screenshot button text to speech (`320f768c`)

## 0.20.0
Released on 2024-07-14.

- Changed version number with v0.20.0 (`624d7c3e`)
- feat: Added get_current_time function (#207) (`8f15bf5d`)
- fix: Resolved input box include bug to llm input (`492fe49d`)
- fix: Resolve circular import (`f324b0a0`)
- fix: Fixed a bug in read_part_task_generate_only (`0349c5ee`)
- refactor: Fixed dictionary iteration by using items() method (`b28955e2`)
- refactor: autofix issues in 2 files (`22f2222a`)
- rafactor: Removed some prints (`96afd5d5`)
- perf: Improved whole text and speech time (5x text, 2x speech) (`ace952fa`)
- refactor: Fixed circular imports for get_tools function (`5fe554f3`)
- fix: Fixed tools (`06f6c4d3`)
- refactor: Removed an unused if (`48260495`)
- refactor: remove unnecessary f-string (#201) (`6ab6c003`)
- refactor: use identity check for comparison to a singleton (#200) (`7aa55728`)
- fix: Fixed tool len increasing bug (`9e2f66bf`)
- Fix: Writing a general wrapper for handling standard_tools section (#199) (`1ef06626`)
- fix: Solved imports (`e1b7fc2c`)
- fix: Changed api interface to localhost (#197) (`136625d7`)
- fix: Version parameter with os.system command usage, added shlex.quote (#196) (`c68ee9bd`)
- refactor: remove reimported module (#195) (`24cefa5b`)
- fix: After reformating (`585f8d54`)
- refactor: remove unused imports (#194) (`72aa2a4e`)
- refactor: add newline at end of file (#193) (`733b4508`)
- refactor: remove unnecessary whitespace (#192) (`af3479a5`)
- ci: add .deepsource.toml (`17efd313`)
- feat: Added offline app close function (#191) (`800165a4`)
- feat: Added offline app open functions (#190) (`6c364815`)
- style: Changed border radiust to half (#187) (`1219e7e0`)
- perf: Improved first setting to make a clean output after code generation (`46710e75`)
- perf: Improved assistant first system message (`9e9a04ee`)
- feat: predefined agents turned on by default (#186) (`b9d10904`)
- feat: Added old code compatbility for core write team (#185) (`0ae8e5a0`)
- feat: Added local python repl tool (#184) (`e06e5b7f`)
- perf: Improved code team code quality (#183) (`50746aa4`)
- fix: Fixed dynamic island inner function usages (`c090af3b`)
- fix: Some fixes for dynamic island (`a3c82e20`)
- docs: Added example use cases (#182) (`215975d5`)
- feat: Added dynamic top bar for showing used tools like Dynamic Island (#181) (`65829183`)

## 0.19.1
Released on 2024-06-20.

- Changed version number with v0.19.1 (`f6099381`)
- feat: Added on of off option for cintinuously conversations (#179) (`6f831479`)

## 0.19.0
Released on 2024-06-20.

- Changed version number with v0.19.0 (`767e1680`)
- feat: Added wait function to remote (`862bf326`)
- feat: Added talk API (#178) (`b3b9151a`)
- docs: Added information new API functions (`a830abf1`)
- fix: Invalid tools array too long (#177) (`78169a43`)
- feat: Added talking ability to api controlled input system for some cases (#175) (`83e60dd5`)
- feat: Added just screenshot api (#174) (`9f50f67b`)
- feat: Added an settings to set wake word operation (#173) (`04fad7a5`)
- Added an information about our new capabilities (`f98c84d4`)
- feat: Added keyboard_press functioanlity (#172) (`68887150`)
- feat: Added keyboard_write functionalitiy (#171) (`2943b06b`)

## 0.18.2
Released on 2024-06-18.

- Changed version number with v0.18.2 (`7de586ec`)
- Fixed standart tools tool is not defined bug (#170) (`0db3316e`)
- Added techstack (`88fa01c0`)
- Added integrations image (`560649b2`)
- Fix: Improve parsing (#151) (`b9890ceb`)

## 0.18.1
Released on 2024-06-18.

- Changed version number with v0.18.1 (`d9c50547`)
- Fixed the_input_box_pre None bug (#169) (`8253aa65`)
- Added examples for all Remote_Client methods to the API documentation. (`8b47c7a5`)

## 0.18.0
Released on 2024-06-17.

- Changed version number with v0.18.0 (`e4169539`)
- Update README.md (`deaa3a7d`)
- Some improvements (`77ffdf55`)
- Feat: Added mouse move and scroll tools (#166) (`457c0d94`)
- feat: Added profile, reset_memory, activate an deactivate predefined_agents, and activate and deactivate online tools apis (#165) (`22a2116e`)
- feat: Added sleep tool for offline tool mode (#164) (`84477414`)
- feat: Added open_url tool in offline tool mode (#163) (`34e12a5f`)

## 0.17.0
Released on 2024-06-16.

- Changed version number with v0.17.0 (`137a3e85`)
- Fixed api /input (`3f016dda`)
- Fixed start and stop mechanism of api (`bebbe46b`)
- Fixed api stop proccess (`9f57b25f`)
- Refactor process.py for better code organization and readability (`59a6c8a0`)
- feat: Added API system (#162) (`b1311227`)
- feat: Added more feedback about assistant status (#161) (`2503fde2`)

## 0.16.7
Released on 2024-06-16.

- Changed version number with v0.16.7 (`7a920fc6`)
- feat: If the user chooses to type text, disable continuously conversation (#158) (`6d8ae5c2`)

## 0.16.6
Released on 2024-06-16.

- Changed version number with v0.16.6 (`6c6bea95`)
- Fixed text input box bug (`12f74ba5`)

## 0.16.5
Released on 2024-06-15.

- Changed version number with v0.16.5 (`568ab761`)
- Fixed bug about system messages (#157) (`dc155cbf`)

## 0.16.4
Released on 2024-06-15.

- Changed version number with v0.16.4 (`cdfab794`)
- Changed default value of online tools (tiger) for faster loading times at first time (#156) (`51d6ffae`)

## 0.16.3
Released on 2024-06-15.

- Changed version number with v0.16.3 (`575be79e`)
- Fixed long response time at first (#155) (`4bcb527d`)

## 0.16.2
Released on 2024-06-15.

- Changed version number with v0.16.2 (`bef36672`)
- Fixed stop talking bug via collapse button (#154) (`5952b2fc`)
- Fixed a bug about stopping when wake word actively working (#153) (`83274094`)
- Fixed a bug about manuel suspend and continuously converstaions (#152) (`7ce1e8b9`)
- Some improvements (`86d2b3f3`)
- Fixed discord link (`0b24bae5`)
- Refactor wake word mechanism for improved performance and reliability (`d05813c6`)

## 0.16.1
Released on 2024-06-15.

- Changed version number with v0.16.1 (`fe08f014`)
- Fixed requirement problem (`acc75874`)
- Fixed tiger tools problem (`7c7afe5c`)
- Refactor wake word mechanism for improved performance and reliability (`bbddf9a6`)
- feat: Added generate_code_with_aim_team for better code generation and copy in predefined agents mode (#150) (`069ae65a`)
- feat: Update TTS function to support non-threaded execution (`d1ae0398`)
- feat: Added intro words for wake word (#149) (`d241c8ff`)
- feat: Stop talking after Wake Word (#148) (`3bcaa708`)

## 0.16.0
Released on 2024-06-15.

- Changed version number with v0.16.0 (`a6cfba47`)
- feat: Added Continuously Conversations mode (#147) (`2a253429`)

## 0.15.0
Released on 2024-06-15.

- Changed version number with v0.15.0 (`291631fe`)
- Added informations about Wake Word and Auto Stop Recording (`d4ae0097`)
- feat: Refactor wake word mechanism for improved performance and reliability (`bb8591e2`)
- Fix for dependencies (`8de608c9`)
- chore: Update requirement_controller.yml to install additional dependencies (`8ba6fdfa`)
- Added an on off option for wake word (`fd947f89`)
- Fixed just screenshot mode (#146) (`eea7d1bf`)
- feat: Adding wake word mechanism (#118) (`e7b0bc37`)
- feat: Added auto recording stop mechanism by microphone threshold, dynamic setting (#145) (`0b440e36`)
- Changed intro png (`3ba32936`)
- Changed usage.png (`fc7e5f6f`)
- Complete collaborated speaking feature and update roadmap (`01bdaea7`)
- feat: Added collaborated speaking (#144) (`3286fc89`)
- chore: Update roadmap and add new voice model options (`d44dbc07`)
- feat: Added exception information via tts (#143) (`edd9f594`)

## 0.14.5
Released on 2024-06-14.

- Changed version number with v0.14.5 (`4bbf02b0`)
- Changed version number with v0.14.4 (`9a35e9e7`)
- Failed native application roadmap (#142) (`306fa0f3`)

## 0.14.4
Released on 2024-06-14.

- Changed version number with v0.14.4 (`61ce8bea`)
- Failed native application roadmap (`1be0b524`)
- chore: Update PyInstaller dependencies and installation process (`c76a8ede`)
- chore: Update PyInstaller dependencies and installation process (`1c83d7c5`)
- chore: Update PyInstaller dependencies and installation process (`325523f2`)
- chore: Update PyInstaller dependencies and installation process (`4a7c54ce`)
- chore: Add embedchain.llm to app.spec and update PyInstaller dependencies (`c9b2911f`)
- chore: Update PyInstaller dependencies and installation process (`6a1585a5`)

## 0.14.3
Released on 2024-06-14.

- Changed version number with v0.14.3 (`1e3c2069`)
- chore: Update PyInstaller dependencies and installation process (`0edac4a1`)
- chore: Update PyInstaller dependencies and installation process (`6fc9ff8f`)
- Added embedchain.llm to app.spec (`5837aa2c`)
- chore: Update PyInstaller dependencies and installation process (`bbe42dbd`)
- chore: Update hookspath and hiddenimports for PyInstaller compatibility (`19755c88`)
- chore: Update hookspath to include 'hooks/' directory (`561481f3`)
- chore: Update hiddenimports to include 'embedchain' and 'crewai' libraries (`a6b69561`)
- chore: Update hiddenimports to include 'embedchain' library (`087c7db9`)
- Fix (`c4270b51`)
- chore: Include crewai library in the application bundle (`4243e0d1`)
- Fix (`8c3609fa`)
- Fix (`17e85cf5`)
- Fix (`a4fdbbea`)
- Fix (`cea68467`)
- Fixed (`43b49f60`)
- Fix (`b5783885`)
- Fix (`24c572e6`)
- Fix try (`4a661966`)
- Fix (`cc868cd8`)
- Fix try (`341cd236`)
- chore: Update requirements file to 'requirements_comb.txt' (`b5449e9e`)
- chore: Update requirements file to 'requirements_comb.txt' (`477efe3f`)
- chore: Add 'crewai' to hiddenimports in app.spec (`0bf6f23f`)
- chore: Update button text for disabling Just Text Model (`0daaf661`)
- Added requirement test system (`50be9844`)
- Added installation links for macos and windows (`1bfd7820`)

## 0.14.2
Released on 2024-06-13.

- Changed version number with v0.14.2 (`de534d7c`)
- Fixed pyqt5 requirement for macos (`2e3f692a`)
- Fix (`0f1d6663`)
- Fix (`0b766f15`)
- Fix (`58f1836e`)
- Fix (`bfb0cabb`)

## 0.14.1
Released on 2024-06-13.

- Changed version number with v0.14.1 (`64a064a7`)
- Fixed requirement pywin32 (`7c31ad1a`)
- Added linux and macos builds (`3cfe7297`)
- Fix (`f98a3072`)
- Fix (`c461a13f`)

## 0.14.0
Released on 2024-06-13.

- Changed version number with v0.14.0 (`c57cf913`)
- Added windows .exe to workflow (`0f6ff818`)
- chore: Refactor search_on_internet_and_report_team_ function (`62ad1c0b`)
- Fixed agentic infra llm context (`9aa6d6e2`)
- feat: Update input box placeholder text in llmsettings popup (`6ba76e6e`)
- Added online tools on off option and a fallback mechanism when they not accassible (`d396e240`)
- Create app.spec (`406317de`)
- chore: Update package.yml (`7cd9f6c2`)
- chore: Remove unnecessary code in settings_popup function (`5eb518a8`)
- Added packaging workflow (#138) (`644462a6`)
- Updated readme (`f40dbc06`)
- feat: Added predefined agents for best quality of results (#133) (`a632199e`)
- Fix (`c2ac4945`)
- Added a space to fix x icon (`8e5e8f39`)
- Added a different color for thinking procces (#132) (`53aa786b`)
- Added a small border (`7c82675b`)
- Fixed long text problem in input box (#131) (`8de68299`)
- feat: Added stop talking mechanism (#130) (`d5f2780d`)
- Fix for collapse button when the assistant is thinking (`71b44e5b`)
- Some fixes for transparency (`e0af77eb`)
- feat: Added auto transparency feature when the mouse not on the app (#128) (`725114cd`)
- feat: Added double click ability (#127) (`3e723acc`)

## 0.13.0
Released on 2024-06-12.

- Changed version number with v0.13.0 (`4d05374e`)
- Removed print in chat history (`c7f99cd6`)
- Fixed input box last response (`a11d0d85`)
- Fix for new view (`fc93fa59`)
- New transparancy and border radius (#126) (`f15c3e48`)
- Added custom title bar (#124) (`56edc408`)
- feat: Added click to an text or icon in the screen abilityy (#122) (`d4e83efd`)
- Added requirements.in for some dependencies (`e185abaa`)
- fix: various typos (#121) (`a2fac85f`)
- Fix for input text adding (`ed46c1fb`)
- Included input box text when buttons are used. (#115) (`832b6010`)

## 0.12.0
Released on 2024-06-10.

- Changed version number with v0.12.0 (`7154bd92`)
- Added LLaVA-Phi-3 (#114) (`82d264b2`)
- Llama3 & Gemini Pro support (#111) (`379a6fc7`)
- Removed debug thing (`ad5bdc5b`)
- Fix for signal.py (`884822fc`)
- Update README.md (`35843f2b`)
- Fix max requests query and wrong import dependency (#112) (`ea428e0d`)

## 0.11.0
Released on 2024-06-09.

- Changed version number with v0.11.0 (`d35b9de2`)
- Many UI improvement and dark mode (#105) (`c3a86ab5`)
- added docstrings for bump.py (#103) (`cdb63e26`)
- Add docstrings to improve code documentation (#102) (`b37699a6`)

## 0.10.0
Released on 2024-06-09.

- Changed version number with v0.10.0 (`c6889a36`)
- Changed font and logo (#98) (`66224847`)
- Added error handling and information (#97) (`9e4503c2`)
- Added gpt-4-turbo support (#96) (`1373124f`)

## 0.9.2
Released on 2024-06-08.

- Changed version number with v0.9.2 (`a5a11e99`)
- Fixed character (#94) (`02198736`)

## 0.9.1
Released on 2024-06-08.

- Changed version number with v0.9.1 (`b944db01`)
- Fixed version number in setup.py (`6e72adbb`)
- Fix (`3a1fda21`)

## 0.9.0
Released on 2024-06-08.

- Changed version number with v0.9.0 (`ed41b4a6`)
- Fixed bump system (`a1c466d2`)
- Update bump.py (`4d975453`)
- Update bump.py (`0ad7e637`)
- Added character to gpt-computer-assistant (#90) (`ec94e933`)
- Increased max retries of openai (`ced13b0f`)
- Fixed +screenshot button bug caused by collapse (`ee7d6046`)
- Added collapse option for ui (#89) (`358d115c`)
- Added more space (`bca7a4fb`)
- Many improvements for input box (#88) (`42f9cd73`)
- Added llm settings system for multiple model supports (#86) (`bf6f25f4`)
- Added gpt-3.5-turbo support (#85) (`6ad75a03`)
- Requirement adpating + unbuild environement + handler for other model and paintEvent (#77) (`295c97cb`)
- First review of the get_model() to maintain easy extensive SW architecture (#83) (`bc76b0ad`)
- Fixed openai url (`08b268f2`)
- Refactor pulse_circle to use modulus for resetting pulse_frame (#67) (`77a71181`)
- Add the function of custom base_url. (#71) (`af6fcf76`)
- Add new MD lang (#72) (`990af107`)
- Pin dependencies version (#73) (`db02b341`)
- feat: add black formatter (#66) (`8ae94e09`)

## 0.8.9
Released on 2024-06-06.

- Changed version number with v0.8.9 (`0174cfb2`)
- Added os_name to telemetry (#65) (`9129d39c`)
- Fixed a typo (`bd830ec0`)
- Added Custom Tools mechanism (#64) (`0527fc1e`)
- Some edits to readme (`6338978b`)

## 0.8.8
Released on 2024-06-06.

- Changed version number with v0.8.8 (`ced8bf9d`)
- Fix (`b0f6a3a2`)
- Fix (`762f068a`)
- Added section about agentic infra (`60c218d3`)
- Fix (`35c65973`)
- Fix (`715063af`)
- Fixed agentic infra (#61) (`b2cc29c1`)

## 0.8.7
Released on 2024-06-06.

- Changed version number with v0.8.7 (`b6afda33`)
- Fix (`870d9c60`)

## 0.8.6
Released on 2024-06-06.

- Changed version number with v0.8.6 (`0b3160ec`)
- Fix (`1c28993b`)
- Added information about agentic (`591566ca`)

## 0.8.5
Released on 2024-06-06.

- Changed version number with v0.8.5 (`f8ce8104`)
- Fixed crewai requirement problem (`c4cc7176`)
- Fixed groq ai message history bug (#59) (`e8a1bf63`)
- Cleaning (`a63958d2`)
- feat: add tracing for the processes and records (#54) (`12b640b2`)
- Fix bug : Removed screenshot_path variable (#57) (`6c3212ac`)
- Minor typo tweaks (#58) (`d03d74de`)
- Moved text to top (`28c7a3d2`)
- Moved demo video to top (`d9c08fcf`)
- Added an information about groq (`374b023d`)
- Added an image (`7fa1f8bd`)
- Adapt the whole code for fast un-building dev environment (#53) (`dd96efaa`)
- Added wide banner (`10d29127`)

## 0.8.4
Released on 2024-06-06.

- Changed version number with v0.8.4 (`e583ea22`)
- Fixed tools (`57a21b68`)

## 0.8.3
Released on 2024-06-06.

- Changed version number with v0.8.3 (`1092e008`)
- UI improvements and fixed window size (#51) (`4b971c37`)

## 0.8.2
Released on 2024-06-06.

- Changed version number with v0.8.2 (`af609707`)
- Fix (`555a0789`)

## 0.8.1
Released on 2024-06-06.

- Changed version number with v0.8.1 (`aa7f2409`)
- Added exception handling in encode_image to manage file (#49) (`099c52b0`)
- Added include_package_data (`84750635`)
- Update MANIFEST.in (`7f8916fa`)
- Fix scaling on high DPI monitors (#47) (`e910b31c`)

## 0.8.0
Released on 2024-06-06.

- Changed version number with v0.8.0 (`db9400de`)
- Fixed some bugs (`02e3543a`)
- Fix (`19cf9887`)
- Updated requirement (`d3371e81`)
- Added groq support (#46) (`0584b15c`)
- Added agentic infrastructure (#45) (`f97efcee`)
- Add infrastack and telemetry integration (#41) (`08c20967`)
- Update input validation for version part argument in main function. (#39) (`c45e7594`)
- Organize import statements (#38) (`01fcbfd9`)

## 0.7.1
Released on 2024-06-05.

- Changed version number with v0.7.1 (`83f5b919`)
- Added new ui screenshot (`c99a0245`)
- Fix typos and improve readability (#36) (`7aacb27d`)
- Added icons (#37) (`a20b4a2a`)
- Edit roadmap (`14037616`)

## 0.7.0
Released on 2024-06-05.

- Changed version number with v0.7.0 (`487ec861`)
- Fixed a bug in python code in the readme (#33) (`08e43de7`)
- Update README.md (`077cc4ed`)
- Update README.md (`c2931357`)
- Added information about local models (`dd67786f`)
- Adding local model support for text+vision capabilities (#19) (`ddb26ddd`)

## 0.6.14
Released on 2024-06-05.

- Changed version number with v0.6.14 (`6292965b`)
- Reduced long loading time (#31) (`bd6143ee`)
- Added openai platform link for geting api key (#30) (`d4fe7d88`)
- Adding information about local model support (`6324370d`)
- Add Simple Chinese README (#28) (`1ec8f071`)
- Added x account (`8a456fb3`)
- Changed demo video (`24e4a3c9`)
- Changed options image (`9695e556`)
- Some edits on readme (#25) (`d1971d24`)
- Update README.md (`9d623d80`)
- Some edits to readme (`b2865ac9`)
- Added information about needed python version (`88bf5f77`)

## 0.6.13
Released on 2024-06-04.

- Changed version number with v0.6.13 (`1529d0f5`)
- Added limited support for 3.9 (`6f4d4e4e`)
- Added another readme (`aa64b508`)

## 0.6.12
Released on 2024-06-04.

- Changed version number with v0.6.12 (`f9db0cce`)
- Added langchain_experimental to requirements.txt (`7e1560cd`)
- Added try except to tiger tools (`d1ff7f29`)
- Small fix (`7470d990`)
- Update README.md (`57114615`)
- Create CONTRIBUTING.md (`6b458e3a`)
- Added a todo (`9d7a4ff5`)
- Removed some typos (`a7b39c84`)
- Update README.md (`8bb10bee`)
- Update README.md (`b04922d1`)

## 0.6.11
Released on 2024-06-03.

- Changed version number with v0.6.11 (`826243e4`)
- Added a requirement (`87a6a8f6`)

## 0.6.10
Released on 2024-06-03.

- Changed version number with v0.6.10 (`ba8acd90`)
- Fix (`3abd06ff`)

## 0.6.9
Released on 2024-06-03.

- Changed version number with v0.6.9 (`669901fc`)
- Added many events (`6f2113fb`)

## 0.6.8
Released on 2024-06-03.

- Changed version number with v0.6.8 (`878ee94e`)
- Fixed settings location (`f3b6f4ff`)

## 0.6.7
Released on 2024-06-03.

- Changed version number with v0.6.7 (`442a1603`)
- Added pydantic to requirements (`13ce9428`)
- Some typo fix (`38b5e0ae`)
- Moved installation to top (`00c74239`)

## 0.6.6
Released on 2024-06-03.

- Changed version number with v0.6.6 (`f171aef3`)
- Update agent.py (`f9a1780a`)
- Fix (`7b9966f4`)

## 0.6.5
Released on 2024-06-03.

- Changed version number with v0.6.5 (`3b685805`)
- Fix requirements (`a9e35a63`)

## 0.6.4
Released on 2024-06-03.

- Changed version number with v0.6.4 (`0004c9da`)
- Fixed ui bugs (`901b519a`)
- Added a requirement (`1c3cbf7d`)

## 0.6.3
Released on 2024-06-03.

- Changed version number with v0.6.3 (`850783f7`)
- Added tool requirements (`56a21213`)

## 0.6.2
Released on 2024-06-03.

- Changed version number with v0.6.2 (`d2c6a951`)
- Added pyscreeze to requirements.txt (`1d0178d1`)
- small typo fix (#16) (`7ff15334`)
- Update gpt_computer_assistant.py (#14) (`95fb355a`)
- Added --profile flag doc (#15) (`3886f23d`)
- Added intel macos information (`5d27ef52`)
- Addded dc (`12e6849c`)
- Added upsonic logo (`64ce27c1`)
- Added upsonic tiger information (`9d7ad2f8`)
- Update README.md (`aaf815bc`)
- Added video demo (`c2482939`)

## 0.6.1
Released on 2024-06-02.

- Changed version number with v0.6.1 (`ce0c41e1`)
- Fix (`6bbf7ee0`)
- Update README.md (`51ccf1c0`)
- Update README.md (`7e8d3110`)
- Update README.md (`d25afd76`)
- Update README.md (`e576b2f6`)

## 0.6.0
Released on 2024-05-29.

- Changed version number with v0.6.0 (`d161139c`)
- Added send text with screenshot button (#8) (`21c57523`)

## 0.5.6
Released on 2024-05-29.

- Changed version number with v0.5.6 (`06a84616`)
- Fix (`1853d346`)

## 0.5.5
Released on 2024-05-29.

- Changed version number with v0.5.5 (`c7f5d105`)
- Minimal feature (`10251521`)

## 0.5.4
Released on 2024-05-29.

- Changed version number with v0.5.4 (`1723625f`)
- Fix (`fcfc9c8a`)

## 0.5.3
Released on 2024-05-29.

- Changed version number with v0.5.3 (`aae17c32`)
- Fix (`b52532ad`)

## 0.5.2
Released on 2024-05-29.

- Changed version number with v0.5.2 (`a93ff7af`)
- Fix (`b0ef7617`)

## 0.5.1
Released on 2024-05-28.

- Changed version number with v0.5.1 (`22756466`)
- Fixed high delay problem (#7) (`8fbc1f9d`)

## 0.5.0
Released on 2024-05-28.

- Changed version number with v0.5.0 (`c4ceb708`)
- Added different profiles mode (#6) (`00ddd47f`)

## 0.4.0
Released on 2024-05-28.

- Changed version number with v0.4.0 (`c7926ae6`)
- Update README.md (`40e60d18`)
- Added just text mode (#5) (`0d82432f`)
- Increased effect speed (`4c06bf6b`)
- Update README.md (`b8ec5c4a`)
- Added a minimal feature todo (`c8b0cac0`)

## 0.3.0
Released on 2024-05-28.

- Changed version number with v0.3.0 (`d967f0e3`)
- Update README.md (`816aa231`)
- Update README.md (`b774aec3`)
- Added icon (#4) (`0fee61e9`)
- Added input box and button (#3) (`456b4288`)
- Added splitting long audios (#2) (`8951b1b0`)

## 0.2.0
Released on 2024-05-27.

- Changed version number with v0.2.0 (`365d6239`)
- Added reset system (#1) (`34fcf02c`)
- Added usecases (`9b320284`)
- Some imrpovements (`8f8aab00`)

## 0.1.2
Released on 2024-05-27.

- Changed version number with v0.1.2 (`d77bb103`)
- Fix (`e8988aec`)

## 0.1.1
Released on 2024-05-27.

- Changed version number with v0.1.1 (`cb203784`)
- Fix (`09a68b4b`)

## 0.1.0
Released on 2024-05-27.

- Changed version number with v0.1.0 (`920866c0`)
- Added pipeline (`e0a409e6`)
- Update README.md (`5f9c46f3`)
- Update README.md (`6fa4a259`)
- A small fix (`61a96482`)
- Added entry point (`688d76ad`)
- Small fix (`66621dd8`)
- Added necessary things (`b79a7def`)
- Initial commit (`bcfb8940`)

---

From version 0.76.2 onward, changelog entries are generated by release-please.
