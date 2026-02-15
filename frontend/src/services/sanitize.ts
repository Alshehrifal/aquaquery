/**
 * Defense-in-depth content sanitizer for chat messages.
 * Strips any internal tool call markup that may have leaked through the backend.
 */

const TOOL_CALL_PATTERNS = [
  // Full blocks with content
  /<tool_call>[\s\S]*?<\/tool_call>/gi,
  /<function_calls>[\s\S]*?<\/function_calls>/gi,
  /<invoke\b[^>]*>[\s\S]*?<\/invoke>/gi,
  /<tool_result>[\s\S]*?<\/tool_result>/gi,
  // Self-closing tags
  /<tool_call\b[^/]*\/\s*>/gi,
  /<function_calls\b[^/]*\/\s*>/gi,
  /<invoke\b[^/]*\/\s*>/gi,
  // Orphan opening/closing tags
  /<\/?tool_call[^>]*>/gi,
  /<\/?function_calls[^>]*>/gi,
  /<\/?invoke[^>]*>/gi,
  /<\/?parameter[^>]*>/gi,
  /<\/?tool_result[^>]*>/gi,
  /<\/?antml:invoke[^>]*>/gi,
  /<\/?antml:parameter[^>]*>/gi,
  /<\/?antml:function_calls[^>]*>/gi,
];

const MULTI_BLANK_LINES = /\n{3,}/g;

export function sanitizeContent(content: string): string {
  if (!content) return content;

  let result = content;
  for (const pattern of TOOL_CALL_PATTERNS) {
    result = result.replace(pattern, '');
  }

  result = result.replace(MULTI_BLANK_LINES, '\n\n');
  return result.trim();
}
