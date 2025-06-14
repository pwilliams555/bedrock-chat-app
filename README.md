# AWS Bedrock Chat Application

A serverless chat application built with AWS Bedrock for AI-powered conversations.

## Architecture

- **Frontend**: HTML/JavaScript chat interface
- **Backend**: Python Lambda function with AWS Bedrock integration
- **Infrastructure**: AWS serverless architecture

## Files

- `frontend/index.html` - Chat interface with modern UI
- `backend/index.py` - Lambda function handling chat requests

## Development Workflow

This project uses automated S3 deployment with GitHub Actions:

### Branches
- `main` → Production: `s3://edgeworth-apps/model-chat/`
- `develop` → Dev site: `s3://edgeworth-apps/model-chat/dev/`
- Feature branches for new work

### Development Process
1. **Make changes** to `frontend/index.html` locally
2. **Commit and push** to develop branch:
   ```bash
   git add frontend/index.html
   git commit -m "Update chat interface"
   git push origin develop
   ```
3. **Auto-deployment** triggers to dev S3 bucket
4. **Test changes** on dev site
5. **Deploy to production** by merging develop → main

### Manual Deployment (if needed)
```bash
# Deploy to dev
aws s3 sync frontend/ s3://edgeworth-apps/model-chat/dev/ --delete

# Deploy to production  
aws s3 sync frontend/ s3://edgeworth-apps/model-chat/ --delete
```

## Getting Started

1. Clone the repository
2. Make changes to `frontend/index.html`
3. Push to `develop` branch for automatic dev deployment
4. Deploy Lambda function to AWS when ready
5. Configure AWS Bedrock permissions

## Features

- Real-time chat interface
- AWS Bedrock AI model integration
- Responsive design
- Error handling and loading states
- Automated S3 deployment via GitHub Actions