# Voice Agent Widget

Embeddable voice assistant widget powered by Gemini Live API for real-time voice conversations and automated appointment booking.

## Project Structure

```text
voice-agent-widget/
├── widget.js          # Main embeddable widget script
├── README.md          # Project documentation
└── widget-host/
    ├── index.html     # Hosted widget status and landing page
    └── test-embed.html# Embed test and verification page
```

## How to Embed

Include the script on any webpage:

```html
<script src="/widget.js" data-client="your-client-id"></script>
```

The widget will inject a floating microphone button in the bottom-right corner. When clicked, it establishes a live audio session with the voice assistant.
