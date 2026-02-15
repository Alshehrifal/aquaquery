"""Tests for response sanitizer."""

from backend.api.sanitizer import sanitize_response


class TestSanitizeResponse:
    def test_strips_tool_call_block(self):
        content = 'Hello <tool_call>query_ocean_data{"variable": "TEMP"}</tool_call> world'
        result = sanitize_response(content)
        assert "<tool_call>" not in result
        assert "query_ocean_data" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strips_function_calls_block(self):
        content = (
            "Some text\n"
            "<function_calls>\n"
            '<invoke name="query_ocean_data">\n'
            '<parameter name="variable">TEMP</parameter>\n'
            "</invoke>\n"
            "</function_calls>\n"
            "More text"
        )
        result = sanitize_response(content)
        assert "<function_calls>" not in result
        assert "<invoke" not in result
        assert "<parameter" not in result
        assert "query_ocean_data" not in result
        assert "Some text" in result
        assert "More text" in result

    def test_strips_orphan_tags(self):
        content = "Before <tool_call> middle </tool_call> after"
        result = sanitize_response(content)
        assert "<tool_call>" not in result
        assert "</tool_call>" not in result
        assert "Before" in result
        assert "after" in result

    def test_strips_self_closing_tags(self):
        content = 'Text <invoke name="test" /> more'
        result = sanitize_response(content)
        assert "<invoke" not in result
        assert "Text" in result
        assert "more" in result

    def test_strips_tool_result_block(self):
        content = "Answer <tool_result>some data</tool_result> here"
        result = sanitize_response(content)
        assert "<tool_result>" not in result
        assert "some data" not in result
        assert "Answer" in result

    def test_preserves_clean_content(self):
        content = "The temperature at 500m is 8.2 degrees C."
        result = sanitize_response(content)
        assert result == content

    def test_preserves_markdown(self):
        content = "## Results\n\n- Temperature: 8.2 C\n- Salinity: 35.1 PSU"
        result = sanitize_response(content)
        assert result == content

    def test_collapses_blank_lines(self):
        content = "Before\n\n\n\n\nAfter"
        result = sanitize_response(content)
        assert result == "Before\n\nAfter"

    def test_empty_string(self):
        assert sanitize_response("") == ""

    def test_none_passthrough(self):
        assert sanitize_response("") == ""

    def test_multiline_tool_call(self):
        content = (
            "I'll fetch the data.\n\n"
            "<tool_call>\n"
            "query_ocean_data\n"
            '{"variable": "TEMP", "lat_min": -60}\n'
            "</tool_call>\n\n"
            "Here are the results."
        )
        result = sanitize_response(content)
        assert "<tool_call>" not in result
        assert "query_ocean_data" not in result
        assert "I'll fetch the data." in result
        assert "Here are the results." in result

    def test_multiple_tool_calls(self):
        content = (
            "<tool_call>first_tool{}</tool_call>"
            " Some text "
            "<tool_call>second_tool{}</tool_call>"
        )
        result = sanitize_response(content)
        assert "<tool_call>" not in result
        assert "first_tool" not in result
        assert "second_tool" not in result
        assert "Some text" in result

    def test_mixed_tags(self):
        content = (
            "Hello\n"
            "<tool_call>data</tool_call>\n"
            "<function_calls><invoke name='x'></invoke></function_calls>\n"
            "World"
        )
        result = sanitize_response(content)
        assert "<tool_call>" not in result
        assert "<function_calls>" not in result
        assert "<invoke" not in result
        assert "Hello" in result
        assert "World" in result

    def test_parameter_tags_stripped(self):
        content = '<parameter name="variable">TEMP</parameter>'
        result = sanitize_response(content)
        assert "<parameter" not in result
        assert "</parameter>" not in result

    def test_real_world_leaky_response(self):
        content = (
            "I'll help you find temperature data at 500m depth in the Atlantic Ocean.\n\n"
            '<tool_call> query_ocean_data { "variables": ["TEMP"], '
            '"lat_min": -60, "lat_max": 60, "lon_min": -80, "lon_max": 0, '
            '"depth_min": 450, "depth_max": 550 } </tool_call>\n\n'
            "The average temperature at 500m in the Atlantic is approximately 8.2 degrees C."
        )
        result = sanitize_response(content)
        assert "<tool_call>" not in result
        assert "query_ocean_data" not in result
        assert "variables" not in result
        assert "8.2 degrees C" in result
