"""
Tool Safety Policies for AI Agent Framework

This module provides LLM-based rules and actions for validating tool safety,
detecting harmful tools and malicious tool calls.
"""

import json
from typing import Optional, Dict, Any
from ..base import RuleBase, ActionBase, Policy
from ..models import PolicyInput, RuleOutput, PolicyOutput


class HarmfulToolRule_LLM(RuleBase):
    """
    LLM-powered rule to detect harmful tools during registration.
    
    This rule analyzes tool name, description, and parameters to identify
    tools that could perform harmful operations like:
    - System manipulation (shutdown, restart, privilege escalation)
    - Data destruction (file deletion, database drops)
    - Network attacks (port scanning, DoS attacks)
    - Security violations (credential theft, encryption bypass)
    """
    
    name = "Harmful Tool Detection Rule (LLM)"
    description = "Uses LLM to detect tools with harmful functionality"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
        
        # Configuration
        self.min_confidence = options.get("min_confidence", 0.6) if options else 0.6
        self.strict_mode = options.get("strict_mode", True) if options else True
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process tool information to detect harmful functionality"""
        
        if not self.text_finder_llm:
            # Fallback to keyword-based detection if no LLM available
            return self._keyword_based_detection(policy_input)
        
        try:
            # Build analysis prompt for LLM
            tool_info = self._extract_tool_info(policy_input)
            analysis_prompt = self._build_analysis_prompt(tool_info)
            
            # Use LLM to analyze tool
            from ..llm.upsonic_llm import UpsonicLLMProvider
            if isinstance(self.text_finder_llm, UpsonicLLMProvider):
                llm = self.text_finder_llm
            else:
                llm = UpsonicLLMProvider(
                    agent_name="Tool Safety Analyzer", 
                    model=self.text_finder_llm
                )
            
            # Get LLM analysis
            analysis_result = llm.analyze_tool_safety(
                tool_info=tool_info,
                analysis_type="HARMFUL_FUNCTIONALITY"
            )
            
            if not analysis_result or "is_harmful" not in analysis_result:
                # LLM didn't return expected format, use fallback
                return self._keyword_based_detection(policy_input)
            
            is_harmful = analysis_result.get("is_harmful", False)
            confidence = analysis_result.get("confidence", 0.0)
            reasons = analysis_result.get("reasons", [])
            threat_categories = analysis_result.get("threat_categories", [])
            
            if is_harmful and confidence >= self.min_confidence:
                return RuleOutput(
                    confidence=confidence,
                    content_type="HARMFUL_TOOL",
                    details=f"LLM detected harmful tool functionality: {', '.join(reasons)}",
                    triggered_keywords=threat_categories
                )
            
            return RuleOutput(
                confidence=0.0,
                content_type="SAFE_TOOL",
                details="Tool appears safe based on LLM analysis"
            )
            
        except Exception as e:
            # On error, fall back to keyword-based detection
            return self._keyword_based_detection(policy_input)
    
    def _extract_tool_info(self, policy_input: PolicyInput) -> Dict[str, Any]:
        """Extract tool information from policy input"""
        
        # Tool info is passed in extra_data
        extra_data = policy_input.extra_data or {}
        
        tool_info = {
            "name": extra_data.get("tool_name", "unknown"),
            "description": extra_data.get("tool_description", ""),
            "parameters": extra_data.get("tool_parameters_schema", {}),
            "metadata": extra_data.get("tool_metadata", {})
        }
        
        return tool_info
    
    def _build_analysis_prompt(self, tool_info: Dict[str, Any]) -> str:
        """Build prompt for LLM analysis (not used directly, handled by UpsonicLLMProvider)"""
        return f"""Analyze this tool for harmful functionality:
        
        Tool Name: {tool_info['name']}
        Description: {tool_info['description']}
        Parameters: {json.dumps(tool_info['parameters'], indent=2)}
        """
    
    def _keyword_based_detection(self, policy_input: PolicyInput) -> RuleOutput:
        """Fallback keyword-based detection when LLM is unavailable"""
        
        tool_info = self._extract_tool_info(policy_input)
        tool_name = tool_info.get("name", "").lower()
        tool_description = tool_info.get("description", "").lower()
        
        # Combine for searching
        combined_text = f"{tool_name} {tool_description}"
        
        # Harmful keyword patterns
        harmful_keywords = [
            # System operations
            "delete", "remove", "destroy", "wipe", "erase", "kill", "terminate",
            "shutdown", "reboot", "restart", "format", "rmdir", "rm -rf",
            
            # Security violations
            "exploit", "hack", "crack", "bypass", "privilege", "escalate",
            "rootkit", "backdoor", "malware", "virus", "trojan",
            
            # Network attacks
            "ddos", "dos attack", "port scan", "brute force", "injection",
            "sql injection", "xss", "csrf",
            
            # Data operations
            "drop database", "drop table", "truncate", "mass delete",
            "recursive delete", "purge", "shred",
            
            # Dangerous commands
            "exec", "eval", "system", "shell", "command injection",
            "arbitrary code", "remote code execution"
        ]
        
        triggered = []
        for keyword in harmful_keywords:
            if keyword in combined_text:
                triggered.append(keyword)
        
        if triggered:
            confidence = min(1.0, len(triggered) * 0.3)
            return RuleOutput(
                confidence=confidence,
                content_type="HARMFUL_TOOL",
                details=f"Detected {len(triggered)} harmful keyword(s) in tool",
                triggered_keywords=triggered
            )
        
        return RuleOutput(
            confidence=0.0,
            content_type="SAFE_TOOL",
            details="No harmful keywords detected"
        )


class MaliciousToolCallRule_LLM(RuleBase):
    """
    LLM-powered rule to detect malicious tool calls before execution.
    
    This rule analyzes the actual arguments being passed to a tool call
    to detect suspicious patterns like:
    - Dangerous file paths (system directories, sensitive files)
    - Suspicious commands or scripts
    - Unusual parameter combinations
    - Privilege escalation attempts
    """
    
    name = "Malicious Tool Call Detection Rule (LLM)"
    description = "Uses LLM to detect malicious tool call arguments"
    language = "en"
    
    def __init__(self, options: Optional[Dict[str, Any]] = None, text_finder_llm=None):
        super().__init__(options, text_finder_llm)
        
        # Configuration
        self.min_confidence = options.get("min_confidence", 0.6) if options else 0.6
        self.strict_mode = options.get("strict_mode", True) if options else True
    
    def process(self, policy_input: PolicyInput) -> RuleOutput:
        """Process tool call arguments to detect malicious patterns"""
        
        if not self.text_finder_llm:
            # Fallback to pattern-based detection
            return self._pattern_based_detection(policy_input)
        
        try:
            # Extract tool call information
            tool_call_info = self._extract_tool_call_info(policy_input)
            
            # Use LLM to analyze
            from ..llm.upsonic_llm import UpsonicLLMProvider
            if isinstance(self.text_finder_llm, UpsonicLLMProvider):
                llm = self.text_finder_llm
            else:
                llm = UpsonicLLMProvider(
                    agent_name="Tool Call Safety Analyzer",
                    model=self.text_finder_llm
                )
            
            # Get LLM analysis
            analysis_result = llm.analyze_tool_safety(
                tool_info=tool_call_info,
                analysis_type="MALICIOUS_CALL"
            )
            
            if not analysis_result or "is_malicious" not in analysis_result:
                return self._pattern_based_detection(policy_input)
            
            is_malicious = analysis_result.get("is_malicious", False)
            confidence = analysis_result.get("confidence", 0.0)
            reasons = analysis_result.get("reasons", [])
            suspicious_args = analysis_result.get("suspicious_args", [])
            
            if is_malicious and confidence >= self.min_confidence:
                return RuleOutput(
                    confidence=confidence,
                    content_type="MALICIOUS_TOOL_CALL",
                    details=f"LLM detected malicious tool call: {', '.join(reasons)}",
                    triggered_keywords=suspicious_args
                )
            
            return RuleOutput(
                confidence=0.0,
                content_type="SAFE_TOOL_CALL",
                details="Tool call appears safe based on LLM analysis"
            )
            
        except Exception as e:
            return self._pattern_based_detection(policy_input)
    
    def _extract_tool_call_info(self, policy_input: PolicyInput) -> Dict[str, Any]:
        """Extract tool call information from policy input"""
        
        extra_data = policy_input.extra_data or {}
        
        tool_call_info = {
            "name": extra_data.get("tool_name", "unknown"),
            "description": extra_data.get("tool_description", ""),
            "arguments": extra_data.get("tool_call_args", {}),
            "call_id": extra_data.get("tool_call_id", ""),
            "parameters_schema": extra_data.get("tool_parameters_schema", {})
        }
        
        return tool_call_info
    
    def _pattern_based_detection(self, policy_input: PolicyInput) -> RuleOutput:
        """Fallback pattern-based detection"""
        
        tool_call_info = self._extract_tool_call_info(policy_input)
        arguments = tool_call_info.get("arguments", {})
        
        # Convert arguments to searchable text
        args_text = json.dumps(arguments, default=str).lower()
        
        # Suspicious patterns
        suspicious_patterns = [
            # Dangerous paths
            "/etc/", "/sys/", "/proc/", "/root/", "/boot/",
            "c:\\windows\\system32", "c:\\windows\\",
            
            # Command injection
            "&&", "||", ";", "|", "`", "$(", "${",
            
            # Dangerous commands
            "rm -rf", "dd if=", "mkfs", "fdisk", "> /dev/",
            
            # Network attacks
            "0.0.0.0", "127.0.0.1", "localhost",
            
            # Suspicious flags
            "--force", "--no-preserve-root", "-rf",
            
            # Encoding/obfuscation
            "base64", "hex", "decode", "decrypt",
            
            # Privilege escalation
            "sudo", "su -", "chmod 777", "chmod +x"
        ]
        
        triggered = []
        for pattern in suspicious_patterns:
            if pattern in args_text:
                triggered.append(pattern)
        
        if triggered:
            confidence = min(1.0, len(triggered) * 0.4)
            return RuleOutput(
                confidence=confidence,
                content_type="MALICIOUS_TOOL_CALL",
                details=f"Detected {len(triggered)} suspicious pattern(s) in arguments",
                triggered_keywords=triggered
            )
        
        return RuleOutput(
            confidence=0.0,
            content_type="SAFE_TOOL_CALL",
            details="No suspicious patterns detected"
        )


class ToolBlockAction(ActionBase):
    """Action to block tool registration or execution with detailed message"""
    
    name = "Tool Block Action"
    description = "Blocks tools with harmful functionality or malicious calls"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute blocking action for harmful tools"""
        
        # Lower threshold to catch keyword-based detection (confidence ~0.3)
        # as well as LLM-based detection (confidence ~0.6+)
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        # Generate appropriate block message based on content type
        if rule_result.content_type == "HARMFUL_TOOL":
            block_message = (
                f"Tool registration blocked: This tool appears to have harmful functionality. "
                f"Details: {rule_result.details}. "
                f"If this is a legitimate tool, please verify its safety or adjust safety policies."
            )
        elif rule_result.content_type == "MALICIOUS_TOOL_CALL":
            block_message = (
                f"Tool execution blocked: This tool call contains suspicious arguments. "
                f"Details: {rule_result.details}. "
                f"Please review the arguments and ensure they are safe."
            )
        else:
            block_message = (
                f"Tool blocked by safety policy. Details: {rule_result.details}"
            )
        
        return self.raise_block_error(block_message)


class ToolBlockAction_LLM(ActionBase):
    """LLM-powered action to block tools with contextual messages"""
    
    name = "Tool Block Action (LLM)"
    description = "Uses LLM to generate appropriate block messages for tools"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Execute LLM-powered blocking action"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Tool safety violation: {rule_result.details}"
        return self.llm_raise_block_error(reason)


class ToolRaiseExceptionAction(ActionBase):
    """Action to raise DisallowedOperation exception for harmful tools"""
    
    name = "Tool Raise Exception Action"
    description = "Raises DisallowedOperation exception for harmful tools"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise exception for harmful tools"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        exception_message = (
            f"DisallowedOperation: Tool safety policy violation. "
            f"{rule_result.details}"
        )
        return self.raise_exception(exception_message)


class ToolRaiseExceptionAction_LLM(ActionBase):
    """LLM-powered action to raise exceptions for harmful tools"""
    
    name = "Tool Raise Exception Action (LLM)"
    description = "Uses LLM to generate appropriate exception messages for tools"
    language = "en"
    
    def action(self, rule_result: RuleOutput) -> PolicyOutput:
        """Raise LLM-generated exception for harmful tools"""
        
        if rule_result.confidence < 0.3:
            return self.allow_content()
        
        reason = f"Tool safety policy violation: {rule_result.details}"
        return self.llm_raise_exception(reason)


# Pre-built Policies

## Harmful Tool Block Policy
# Standard policy that blocks harmful tools using keyword + LLM detection
HarmfulToolBlockPolicy = Policy(
    name="Harmful Tool Block Policy",
    description="Blocks tools with harmful functionality during registration",
    rule=HarmfulToolRule_LLM(),
    action=ToolBlockAction()
)

## Harmful Tool Block Policy LLM
# Enhanced policy using LLM for both detection and blocking messages
HarmfulToolBlockPolicy_LLM = Policy(
    name="Harmful Tool Block Policy (LLM)",
    description="Uses LLM for both detection and blocking of harmful tools",
    rule=HarmfulToolRule_LLM(),
    action=ToolBlockAction_LLM()
)

## Harmful Tool Raise Exception Policy
# Policy that raises exceptions for harmful tools
HarmfulToolRaiseExceptionPolicy = Policy(
    name="Harmful Tool Raise Exception Policy",
    description="Raises DisallowedOperation exception for harmful tools",
    rule=HarmfulToolRule_LLM(),
    action=ToolRaiseExceptionAction()
)

## Harmful Tool Raise Exception Policy LLM
# Policy that uses LLM to generate contextual exception messages
HarmfulToolRaiseExceptionPolicy_LLM = Policy(
    name="Harmful Tool Raise Exception Policy (LLM)",
    description="Raises DisallowedOperation with LLM-generated message for harmful tools",
    rule=HarmfulToolRule_LLM(),
    action=ToolRaiseExceptionAction_LLM()
)

## Malicious Tool Call Block Policy
# Policy that blocks malicious tool calls before execution
MaliciousToolCallBlockPolicy = Policy(
    name="Malicious Tool Call Block Policy",
    description="Blocks tool calls with suspicious or malicious arguments",
    rule=MaliciousToolCallRule_LLM(),
    action=ToolBlockAction()
)

## Malicious Tool Call Block Policy LLM
# Enhanced policy using LLM for both detection and blocking
MaliciousToolCallBlockPolicy_LLM = Policy(
    name="Malicious Tool Call Block Policy (LLM)",
    description="Uses LLM for both detection and blocking of malicious tool calls",
    rule=MaliciousToolCallRule_LLM(),
    action=ToolBlockAction_LLM()
)

## Malicious Tool Call Raise Exception Policy
# Policy that raises exceptions for malicious tool calls
MaliciousToolCallRaiseExceptionPolicy = Policy(
    name="Malicious Tool Call Raise Exception Policy",
    description="Raises DisallowedOperation exception for malicious tool calls",
    rule=MaliciousToolCallRule_LLM(),
    action=ToolRaiseExceptionAction()
)

## Malicious Tool Call Raise Exception Policy LLM
# Policy that uses LLM to generate exception messages
MaliciousToolCallRaiseExceptionPolicy_LLM = Policy(
    name="Malicious Tool Call Raise Exception Policy (LLM)",
    description="Raises DisallowedOperation with LLM-generated message for malicious tool calls",
    rule=MaliciousToolCallRule_LLM(),
    action=ToolRaiseExceptionAction_LLM()
)
