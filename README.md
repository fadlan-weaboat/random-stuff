# Advanced Chatbot with SQLite Database

## Overview
This project implements an advanced chatbot that uses:
- Mistral 7B model with 4-bit quantization (when possible)
- XPU acceleration (with CPU fallback)
- SQLite database for persistent storage of user profiles and chat history
- Proper relational database design with foreign key constraints

## Features

### 1. AI Model Integration
- Uses Mistral 7B model for natural language understanding and generation
- Attempts to load model with 4-bit quantization using bitsandbytes for memory efficiency
- Falls back to bfloat16 or default loading if quantization fails
- Utilizes XPU acceleration when available (Intel GPUs), with CPU fallback
- Implements proper response generation with configurable parameters (temperature, top-p, etc.)

### 2. Database Design
The implementation uses SQLite with two main tables:

#### Master Table (User Profiles)
```sql
CREATE TABLE IF NOT EXISTS master (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_name TEXT NOT NULL,
    region TEXT,
    user_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

#### Chats Table (Conversation History)
```sql
CREATE TABLE IF NOT EXISTS chats (
    chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES master(user_id)
)
```

#### Performance Optimization
- Index on `chats(user_id)` for faster query performance

### 3. Core Functionality

#### User Management
- `get_or_create_user()`: Retrieves existing user or creates new one
- Prevents duplicate user entries based on name/region/type combination

#### Chat History Management
- `save_chat_message()`: Stores message-response pairs with timestamp
- `get_chat_history()`: Retrieves chat history with configurable limit
- `get_user_stats()`: Provides user statistics including message counts and timestamps

#### Database Initialization
- Automatic table creation if they don't exist
- Proper foreign key constraints to maintain data integrity

### 4. Chatbot Features
- Interactive command-line interface
- Special commands:
  - `quit`: Exit the chatbot
  - `history`: View recent chat history
- Demo mode with predefined responses when model loading fails
- Error handling for graceful degradation
- User statistics display at startup and shutdown

## Files in This Repository

1. `final_chatbot_with_db.py` - Main implementation with all features
2. `test_final_chatbot.py` - Comprehensive test suite for database functionality
3. `check_db.py` - Utility to examine database contents
4. `demo_chatbot.py` - Simplified demo version
5. Various intermediate versions showing development progression

## Database Schema Benefits

### Normalization
- Separates user profile data from chat messages
- Eliminates data duplication (user info stored once, referenced by ID)
- Ensures data consistency through foreign key constraints

### Query Efficiency
- Indexed foreign key for fast retrieval of user's chat history
- Timestamp ordering for chronological message retrieval
- Flexible LIMIT clauses for pagination

### Data Integrity
- Foreign key prevents orphaned chat records
- NOT NULL constraints on essential fields
- Default timestamps for automatic time tracking

## Usage

### Running the Full Chatbot
```bash
python final_chatbot_with_db.py
```

### Running Tests
```bash
python test_final_chatbot.py
```

### Checking Database Contents
```bash
python check_db.py
```

## Implementation Notes

### Model Loading Strategy
The implementation attempts multiple loading strategies in order:
1. 4-bit quantization with bitsandbytes (most memory efficient)
2. Bfloat16 without quantization (balanced performance/quality)
3. Default loading (maximum compatibility)

This ensures the chatbot works in various environments while optimizing for performance when possible.

### Error Handling
- Graceful degradation to demo mode if model loading fails
- Comprehensive exception handling throughout
- Informative error messages for troubleshooting
- Keyboard interrupt handling for clean exit

### Security Considerations
- Parameterized SQL queries to prevent injection
- Input validation through proper typing
- No external network dependencies beyond initial model download

## Future Enhancements

1. **Web Interface**: Add Flask/FastAPI backend with HTML/JavaScript frontend
2. **Enhanced NLP**: Add intent recognition and entity extraction
3. **Multi-user Support**: Implement proper authentication and session management
4. **Analytics**: Add usage tracking and conversation analytics
5. **Model Management**: Allow switching between different models
6. **Persistence Options**: Add support for other databases (PostgreSQL, MySQL)
7. **Deployment**: Docker containerization for easy deployment

## Conclusion
This implementation demonstrates a robust, production-ready chatbot architecture that combines modern AI capabilities with reliable data persistence. The modular design allows for easy extension and maintenance, while the comprehensive testing ensures reliability.