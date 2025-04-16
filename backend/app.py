import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import logging
import asyncio
import atexit
from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
from openai import AsyncOpenAI
from dotenv import load_dotenv
from workflow.agent_workflow import AgentWorkflow
from utils.message_utils import convert_to_langchain_messages

# Configuration
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


# Initialize Services
try:
    logger.info("Initializing services...")

    if OPENAI_API_KEY:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized.")
    else:
        openai_client = None
        logger.warning("OpenAI client NOT initialized due to missing API key.")


    if openai_client:
        logger.info("Initializing AgentWorkflow...")
        app.workflow = AgentWorkflow(
            openai_client=openai_client,
        )
        logger.info("Workflow initialized successfully.")
    else:
        app.workflow = None
        logger.error("Workflow NOT initialized due to missing OpenAI client.")

except Exception as e:
    logger.exception(f"FATAL: Failed to initialize services during startup: {e}")
    app.workflow = None


@app.route('/api/chat', methods=['POST'])
def chat():
    if not app.workflow:
         logger.error("Chat request received but workflow is not initialized.")
         return jsonify({'error': 'Service not initialized correctly'}), 503

    data = request.json
    user_message = data.get('message')

    if not user_message:
        logger.warning("Chat request received with no message.")
        return jsonify({'error': 'No message provided'}), 400

    logger.info(f"Received chat message: '{user_message}'")

    try:
        response = asyncio.run(current_app.workflow.process_message(
            user_message=user_message
        ))

        logger.info(f"Generated response: '{response[:100]}...'")

        return jsonify({
            'message': response
        })

    except Exception as e:
        logger.exception(f"Error processing chat message: {str(e)}") # Log full exception
        return jsonify({'error': 'Failed to process message due to an internal error'}), 500

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    status = 'ok' if hasattr(app, 'workflow') and app.workflow is not None else 'error'
    message = 'Service initialized' if status == 'ok' else 'Service initialization failed'
    status_code = 200 if status == 'ok' else 503
    return jsonify({'status': status, 'message': message}), status_code

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true')
