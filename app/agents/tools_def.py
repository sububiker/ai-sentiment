SYSTEM_PROMPT = (
    "You are an expert sentiment analysis agent. "
    "Analyze the sentiment of any text, URL, or topic the user provides.\n\n"
    "Guidelines:\n"
    "- If the input contains a URL, call fetch_url to retrieve the content first\n"
    "- If the text is ambiguous or you need broader context, call search_web\n"
    "- If the user asks about trends or past results, call analyze_history\n"
    "- Use only the tools you actually need — for clear plain text, respond directly\n"
    "- You may call multiple tools in sequence if needed\n\n"
    "Always finish with a JSON object and nothing else:\n"
    "{\n"
    '  "label": "positive" or "negative",\n'
    '  "score": float 0.0-1.0,\n'
    '  "reasoning": "one sentence citing the key signal",\n'
    '  "source": "text" | "url" | "web_search" | "combined"\n'
    "}"
)

TOOLS = [
    {
        "name": "fetch_url",
        "description": (
            "Fetch and extract readable text from a URL. "
            "Use this whenever the user provides a link to analyze."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"}
            },
            "required": ["url"],
        },
    },
    {
        "name": "search_web",
        "description": (
            "Search the web for reviews, news, or opinions about a topic. "
            "Use this to gather external context when the input alone is insufficient."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "analyze_history",
        "description": (
            "Look up past sentiment predictions stored in the database for a topic. "
            "Use this when the user asks about trends or historical sentiment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Keyword or topic to look up"}
            },
            "required": ["topic"],
        },
    },
]
