
// Add global fetch for Node.js environment
global.fetch = require('node-fetch');
global.Headers = require('node-fetch').Headers;
global.Request = require('node-fetch').Request;
global.Response = require('node-fetch').Response;

// Run the GitHub MCP server
const server = require('@modelcontextprotocol/server-github');
server.run({ token: process.env.GITHUB_TOKEN });
            