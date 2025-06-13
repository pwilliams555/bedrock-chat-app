import json
import boto3
import logging
import time
import os
import uuid
from decimal import Decimal
from datetime import datetime

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
bedrock_client = boto3.client('bedrock-runtime', region_name="us-east-1")
dynamodb = boto3.resource('dynamodb')
chat_history_table = dynamodb.Table('ChatHistory')
feedback_table = dynamodb.Table('UserFeedback')  # Add feedback table

# Custom JSON encoder to handle Decimal types
class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super(DecimalEncoder, self).default(obj)

def save_message(chat_id, role, content, user_email, model=None, user_name=None, kb_id=None):
    """Save a message to DynamoDB"""
    try:
        timestamp = int(time.time() * 1000)
        item = {
            'chat_id': chat_id,
            'message_id': f"{timestamp}",
            'role': role,
            'content': content,
            'timestamp': timestamp,
            'model': model,
            'user_email': user_email,
            'user_name': user_name or user_email,
            'kb_id': kb_id  # Add KB ID to the stored message
        }
        chat_history_table.put_item(Item=item)
        return True
    except Exception as e:
        logger.error(f"Error saving message: {str(e)}")
        return False

def get_chat_history(chat_id, user_email=None, limit=10):
    """Retrieve recent chat history for a given chat_id"""
    try:
        # If user_email is provided, verify ownership
        if user_email:
            response = chat_history_table.query(
                KeyConditionExpression='chat_id = :chat_id',
                FilterExpression='user_email = :user_email',
                ExpressionAttributeValues={
                    ':chat_id': chat_id,
                    ':user_email': user_email
                },
                ScanIndexForward=False,
                Limit=limit
            )
        else:
            response = chat_history_table.query(
                KeyConditionExpression='chat_id = :chat_id',
                ExpressionAttributeValues={':chat_id': chat_id},
                ScanIndexForward=False,
                Limit=limit
            )
        messages = sorted(response.get('Items', []), key=lambda x: x['timestamp'])
        return messages
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        return []

def get_chat_list(user_email=None, limit=20):
    """Get list of recent chats for a specific user"""
    try:
        if not user_email:
            return []

        logger.info(f"Getting chat list for user: {user_email}")

        # Get chats for specific user with pagination to handle large datasets
        items = []
        scan_kwargs = {
            'FilterExpression': 'user_email = :user_email',
            'ProjectionExpression': 'chat_id, #ts, content, #r, user_name',
            'ExpressionAttributeNames': {
                '#ts': 'timestamp',
                '#r': 'role'
            },
            'ExpressionAttributeValues': {
                ':user_email': user_email
            }
        }
        
        # Handle pagination
        while True:
            response = chat_history_table.scan(**scan_kwargs)
            items.extend(response.get('Items', []))
            
            # Check if there are more items to scan
            if 'LastEvaluatedKey' not in response:
                break
            scan_kwargs['ExclusiveStartKey'] = response['LastEvaluatedKey']
        
        logger.info(f"DynamoDB scan returned {len(items)} total items across all pages")
        
        # Group by chat_id and find the most recent message
        chats = {}
        for item in items:
            try:
                chat_id = item['chat_id']
                timestamp = item['timestamp']
                
                # Handle different content types safely
                content = item.get('content', '')
                if isinstance(content, dict):
                    # Handle image messages or other complex content
                    if content.get('type') == 'image_message':
                        display_text = content.get('text', 'Image message')
                        if not display_text.strip():
                            display_text = 'Image shared'
                    else:
                        display_text = str(content)
                else:
                    display_text = str(content)
                
                # Create preview text
                if len(display_text) > 100:
                    last_message = display_text[:100] + '...'
                else:
                    last_message = display_text
                
                if chat_id not in chats or timestamp > chats[chat_id]['timestamp']:
                    chats[chat_id] = {
                        'chat_id': chat_id,
                        'timestamp': timestamp,
                        'last_message': last_message,
                        'role': item['role'],
                        'user_name': item.get('user_name', user_email)
                    }
                    logger.info(f"Added/Updated chat {chat_id} with timestamp {timestamp}")
                    
            except Exception as e:
                logger.error(f"Error processing chat item {item}: {str(e)}")
        
        # Sort by timestamp (most recent first) and limit
        sorted_chats = sorted(chats.values(), key=lambda x: x['timestamp'], reverse=True)[:limit]
        
        logger.info(f"Returning {len(sorted_chats)} unique chats for user {user_email}")
        logger.info(f"Chat IDs: {[chat['chat_id'] for chat in sorted_chats]}")
        
        return sorted_chats
        
    except Exception as e:
        logger.error(f"Error retrieving chat list: {str(e)}")
        return []

def delete_chat(chat_id, user_email=None):
    """Delete all messages for a given chat_id (only if owned by user)"""
    try:
        # Query all items for this chat_id
        response = chat_history_table.query(
            KeyConditionExpression='chat_id = :chat_id',
            FilterExpression='user_email = :user_email' if user_email else None,
            ExpressionAttributeValues={
                ':chat_id': chat_id,
                ':user_email': user_email
            } if user_email else {':chat_id': chat_id}
        )
        
        # Delete each item
        for item in response['Items']:
            chat_history_table.delete_item(
                Key={
                    'chat_id': item['chat_id'],
                    'message_id': item['message_id']
                }
            )
        
        return True
    except Exception as e:
        logger.error(f"Error deleting chat {chat_id}: {str(e)}")
        return False

def get_knowledge_bases():
    """Get list of knowledge bases"""
    try:
        # Initialize the Bedrock Agent client
        client = boto3.client('bedrock-agent', region_name='us-east-1')
        logger.info("Created Bedrock Agent client")
        
        # Retrieve the list of knowledge bases
        response = client.list_knowledge_bases()
        logger.info(f"Raw response: {response}")
        
        knowledge_bases = []
        for kb in response.get('knowledgeBaseSummaries', []):
            knowledge_bases.append({
                'id': kb['knowledgeBaseId'],
                'name': kb.get('name', kb['knowledgeBaseId']),
                'description': kb.get('description', ''),
                'status': kb.get('status', ''),
                'updated_at': kb.get('updatedAt', '').isoformat() if isinstance(kb.get('updatedAt'), datetime) else ''
            })
        
        logger.info(f"Found knowledge bases: {json.dumps(knowledge_bases)}")
        return knowledge_bases
        
    except Exception as e:
        logger.error(f"Error retrieving knowledge bases: {str(e)}")
        return []

def handle_feedback(event, headers):
    """Handle feedback-related actions"""
    try:
        if event.get('httpMethod') == 'POST':
            body = json.loads(event['body'])
            if body.get('action') == 'submit_feedback':
                feedback_id = str(uuid.uuid4())
                feedback_data = {
                    'id': feedback_id,
                    'timestamp': body.get('timestamp', datetime.utcnow().isoformat()),
                    'user_email': body.get('user_email', 'anonymous'),
                    'name': body.get('name', 'Anonymous'),
                    'message': body.get('message')
                }
                feedback_table.put_item(Item=feedback_data)
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({
                        'success': True,
                        'feedback_id': feedback_id
                    })
                }
        elif event.get('httpMethod') == 'GET':
            query_params = event.get('queryStringParameters', {}) or {}
            if query_params.get('action') == 'list_feedback':
                response = feedback_table.scan()
                feedback_items = response.get('Items', [])
                # Sort by timestamp descending (newest first)
                feedback_items.sort(key=lambda x: x['timestamp'], reverse=True)
                return {
                    'statusCode': 200,
                    'headers': headers,
                    'body': json.dumps({
                        'feedback': feedback_items
                    }, cls=DecimalEncoder)
                }
    except Exception as e:
        logger.error(f"Error handling feedback: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': f'Failed to process feedback: {str(e)}'
            })
        }

def lambda_handler(event, context):
    # Common headers for CORS
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
    }

    try:
        # Handle feedback-related actions
        query_params = event.get('queryStringParameters', {}) or {}
        if query_params.get('action') == 'list_feedback' or (
            event.get('httpMethod') == 'POST' and 
            json.loads(event.get('body', '{}')).get('action') == 'submit_feedback'
        ):
            return handle_feedback(event, headers)

        # Check if this is a knowledge bases request
        if event.get('resource', '').endswith('/knowledge-bases'):
            kbs = get_knowledge_bases()
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({"knowledgeBases": kbs})
            }

        # Handle chat-related requests
        if event.get('httpMethod') == 'GET':
            query_params = event.get('queryStringParameters', {}) or {}
            user_email = query_params.get('user_email')
            
            # If requesting chat list
            if query_params.get('action') == 'list_chats':
                chats = get_chat_list(user_email=user_email)
                return {
                    "statusCode": 200,
                    "headers": headers,
                    "body": json.dumps({"chats": chats}, cls=DecimalEncoder)
                }

            # If requesting specific chat history
            chat_id = query_params.get('chat_id')
            if not chat_id:
                return {
                    "statusCode": 400,
                    "headers": headers,
                    "body": json.dumps({"error": "chat_id is required"})
                }
            
            history = get_chat_history(chat_id, user_email=user_email)
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({"messages": history}, cls=DecimalEncoder)
            }

        # Handle DELETE request
        if event.get('httpMethod') == 'DELETE':
            query_params = event.get('queryStringParameters', {}) or {}
            chat_id = query_params.get('chat_id')
            user_email = query_params.get('user_email')
            
            if not chat_id:
                return {
                    "statusCode": 400,
                    "headers": headers,
                    "body": json.dumps({"error": "chat_id is required"})
                }
            
            success = delete_chat(chat_id, user_email)
            return {
                "statusCode": 200 if success else 500,
                "headers": headers,
                "body": json.dumps({
                    "success": success,
                    "message": "Chat deleted successfully" if success else "Failed to delete chat"
                })
            }

        # Handle POST request for new messages
        if event.get('httpMethod') == 'POST':
            body = json.loads(event['body'])
            message = body['message']
            model_id = body.get('model', 'anthropic.claude-3-7-sonnet-20250219-v1:0')
            kb_id = body.get('knowledgeBaseId', None)
            user_email = body.get('user_email')
            user_name = body.get('user_name')
            test_mode = body.get('test_mode', False)  # Get test_mode flag
            image_data = body.get('image')  # Get image data if present
            
            if not user_email:
                return {
                    "statusCode": 400,
                    "headers": headers,
                    "body": json.dumps({"error": "user_email is required"})
                }
            
            # Only generate chat_id and save messages if not in test mode
            chat_id = body.get('chat_id')
            if not test_mode:
                if not chat_id:
                    chat_id = str(uuid.uuid4())
                    logger.info(f"Generated new chat_id: {chat_id}")

                # Save user message with user info and image if present
                user_content = message
                if image_data:
                    # Store the full image data in the chat history
                    user_content = {
                        "type": "image_message",
                        "image": image_data,
                        "text": message if message else ""
                    }
                save_message(chat_id, "user", user_content, user_email, model_id, user_name, kb_id)

                # Get chat history for context with smart sizing
                chat_history = get_chat_history(chat_id, user_email=user_email, limit=10)
                
                # Reduce context if messages are too long to prevent timeouts
                def get_content_length(msg):
                    content = msg.get('content', '')
                    if isinstance(content, dict):
                        # Handle image messages or other complex content
                        text_content = content.get('text', '') + content.get('image', '')
                        return len(str(text_content))
                    return len(str(content))
                
                total_context_length = sum(get_content_length(msg) for msg in chat_history)
                if total_context_length > 15000:  # If context is too large
                    chat_history = chat_history[-3:]  # Only keep last 3 messages
                    logger.info(f"Reduced context due to size: {total_context_length} chars, using {len(chat_history)} messages")
                elif total_context_length > 8000:  # If context is large
                    chat_history = chat_history[-5:]  # Only keep last 5 messages
                    logger.info(f"Reduced context due to size: {total_context_length} chars, using {len(chat_history)} messages")
                
                logger.info(f"Using {len(chat_history)} messages for context ({total_context_length} chars)")
            else:
                # In test mode, don't use chat history
                chat_history = []
                logger.info("Test mode: skipping chat history")

            # Make the API call
            if kb_id and kb_id != "none":
                client = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
                
                # Handle model ARN format
                model_arn = model_id if "inference-profile" in model_id else f'arn:aws:bedrock:us-east-1::foundation-model/{model_id}'
                logger.info(f"Using model ARN: {model_arn}")

                # Format chat history for KB context with length management
                if chat_history:
                    # Check total context length and reduce if needed
                    context_parts = [f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in chat_history]
                    context = "\n\n".join(context_parts)
                    
                    # If context is too large for KB, reduce it further
                    if len(context) > 10000:
                        # Use only last 2 messages for KB context
                        reduced_history = chat_history[-2:] if len(chat_history) > 2 else chat_history
                        context = "\n\n".join([
                            f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                            for msg in reduced_history
                        ])
                        logger.info(f"Reduced KB context to {len(context)} chars with {len(reduced_history)} messages")
                else:
                    context = ""
                    
                context += f"\n\nHuman: {message}"
                
                response = client.retrieve_and_generate(
                    input={'text': context},
                    retrieveAndGenerateConfiguration={
                        'type': 'KNOWLEDGE_BASE',
                        'knowledgeBaseConfiguration': {
                            'knowledgeBaseId': kb_id,
                            'modelArn': model_arn
                        }
                    }
                )
                ai_response = response.get('output', {}).get('text', 'No response received')
            else:
                # Format messages based on model type
                if "claude-3" in model_id.lower():
                    messages = []
                    for msg in chat_history:
                        content = []
                        if isinstance(msg["content"], dict) and msg["content"].get("type") == "image_message":
                            # Handle image message
                            if msg["content"].get("image"):
                                content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/jpeg",
                                        "data": msg["content"]["image"].split(",")[1] if "," in msg["content"]["image"] else msg["content"]["image"]
                                    }
                                })
                            if msg["content"].get("text"):
                                content.append({
                                    "type": "text",
                                    "text": msg["content"]["text"]
                                })
                        else:
                            # Handle regular text message
                            content.append({
                                "type": "text",
                                "text": msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
                            })
                        
                        messages.append({
                            "role": "user" if msg["role"] == "user" else "assistant",
                            "content": content
                        })
                    
                    # Add current message with image if present
                    current_message_content = []
                    if image_data:
                        # Remove the data:image/... prefix if present
                        base64_image = image_data.split(',')[1] if ',' in image_data else image_data
                        current_message_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        })
                    
                    if message:
                        current_message_content.append({
                            "type": "text",
                            "text": message
                        })
                    
                    messages.append({
                        "role": "user",
                        "content": current_message_content
                    })
                    
                    # Calculate dynamic max_tokens based on context size
                    # Define content length helper if not available
                    def safe_get_content_length(msg):
                        content = msg.get('content', '')
                        if isinstance(content, dict):
                            text_content = content.get('text', '') + content.get('image', '')
                            return len(str(text_content))
                        return len(str(content))
                    
                    # Calculate total context length for Claude 3 models
                    total_claude_context = sum(safe_get_content_length(msg) for msg in chat_history) + len(message)
                    estimated_input_tokens = total_claude_context // 4  # Rough estimate: 4 chars per token
                    max_safe_output_tokens = min(4000, max(1000, 8000 - estimated_input_tokens))
                    logger.info(f"Claude 3 context: {total_claude_context} chars (~{estimated_input_tokens} tokens), Output limit: {max_safe_output_tokens}")
                    
                    request_body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_safe_output_tokens,
                        "messages": messages
                    }
                elif model_id.startswith('amazon.nova'):
                    if image_data:
                        return {
                            "statusCode": 400,
                            "headers": headers,
                            "body": json.dumps({"error": "Image input is not supported by Nova models"})
                        }
                    messages = [
                        {
                            "role": "user" if msg["role"] == "user" else "assistant",
                            "content": [{"text": msg["content"]}]
                        }
                        for msg in chat_history
                    ]
                    # Add current message
                    messages.append({
                        "role": "user",
                        "content": [{"text": message}]
                    })
                    
                    # Calculate dynamic max_tokens for Nova models
                    # Calculate total context length for Nova models  
                    total_nova_context = sum(len(str(msg.get('content', ''))) for msg in chat_history) + len(message)
                    estimated_input_tokens = total_nova_context // 4
                    
                    # Set base limits per Nova model type
                    if 'micro' in model_id:
                        base_max_tokens = min(2000, max(500, 4000 - estimated_input_tokens))
                    elif 'lite' in model_id:
                        base_max_tokens = min(4000, max(1000, 8000 - estimated_input_tokens))
                    elif 'pro' in model_id:
                        base_max_tokens = min(8000, max(1000, 12000 - estimated_input_tokens))
                    else:
                        base_max_tokens = min(4000, max(1000, 8000 - estimated_input_tokens))
                    
                    logger.info(f"Nova context: {total_nova_context} chars (~{estimated_input_tokens} tokens), Output limit: {base_max_tokens}")
                    
                    request_body = {
                        "schemaVersion": "messages-v1",
                        "messages": messages,
                        "inferenceConfig": {
                            "maxTokens": base_max_tokens,
                            "topP": 0.9,
                            "temperature": 0.7
                        }
                    }
                else:
                    # Format history for classic Claude format with length management
                    if chat_history:
                        context_parts = [f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}" for msg in chat_history]
                        context = "\n\n".join(context_parts)
                        
                        # If context is too large, reduce it
                        if len(context) > 12000:
                            reduced_history = chat_history[-2:]
                            context = "\n\n".join([
                                f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                                for msg in reduced_history
                            ])
                            logger.info(f"Reduced classic Claude context to {len(context)} chars")
                    else:
                        context = ""
                    # Calculate dynamic max_tokens for classic Claude
                    estimated_input_tokens = len(context + message) // 4
                    max_safe_output_tokens = min(4000, max(1000, 8000 - estimated_input_tokens))
                    logger.info(f"Classic Claude context: {len(context)} chars (~{estimated_input_tokens} tokens), Output limit: {max_safe_output_tokens}")
                    
                    request_body = {
                        "prompt": f"{context}\n\nHuman: {message}\n\nAssistant:",
                        "max_tokens_to_sample": max_safe_output_tokens,
                        "temperature": 0.7,
                        "stop_sequences": ["\n\nHuman:"]
                    }

                logger.info(f"Invoking model {model_id}")
                start_time = time.time()
                response = bedrock_client.invoke_model(
                    body=json.dumps(request_body),
                    modelId=model_id
                )
                end_time = time.time()
                logger.info(f"Model invocation took {end_time - start_time:.2f} seconds")
                response_body = json.loads(response['body'].read())
                
                # Parse response based on model
                if "claude-3" in model_id.lower():
                    if 'content' in response_body and isinstance(response_body['content'], list):
                        for content in response_body['content']:
                            if content.get('type') == 'text':
                                ai_response = content.get('text', '')
                                break
                        else:
                            ai_response = "No text content found in response"
                    else:
                        ai_response = "Error parsing response"
                elif model_id.startswith('amazon.nova'):
                    if 'output' in response_body and 'message' in response_body['output']:
                        ai_response = response_body['output']['message']['content'][0]['text']
                    else:
                        ai_response = 'Unexpected Nova response format'
                else:
                    ai_response = response_body.get("completion", "No response received")

            # Save assistant's response with KB info
            if not test_mode:
                save_message(chat_id, "assistant", ai_response, user_email, model_id, user_name, kb_id)
            else:
                # Save assistant's response without KB info
                save_message(chat_id, "assistant", ai_response, user_email, model_id, user_name)

            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps({
                    "response": ai_response,
                    "chat_id": chat_id if not test_mode else None
                }, cls=DecimalEncoder)
            }

        # Handle unsupported methods
        return {
            "statusCode": 405,
            "headers": headers,
            "body": json.dumps({"error": "Method not allowed"})
        }

    except Exception as e:
        logger.error("Error in Lambda: %s", str(e), exc_info=True)
        return {
            "statusCode": 500,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": json.dumps({"error": f"Error processing request: {str(e)}"})
        }