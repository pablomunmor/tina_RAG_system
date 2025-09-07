import os
import json
import datetime
import asyncio
import time
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename

# LangChain imports
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI

# Configuration
try:
    from config import HUGGING_FACE_HUB_API_TOKEN, OPENAI_API_KEY
    print("✓ Configuration loaded successfully")
except ImportError:
    HUGGING_FACE_HUB_API_TOKEN = None
    OPENAI_API_KEY = None
    print("⚠️  Warning: config.py not found. Using fallback responses.")

# Constants
UPLOAD_FOLDER = 'uploaded_files'
VECTOR_STORE_PATH = "faiss_index"
LOGS_FOLDER = 'conversation_logs'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# App Setup
app = Flask(__name__)
app.secret_key = 'clementina_health_2024'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure required folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(LOGS_FOLDER, exist_ok=True)

# Enhanced knowledge base with conversational, supportive tone
CLEMENTINA_KNOWLEDGE = {
    "newborn_safety": {
        "keywords": ["blanket", "pillow", "crib", "sleep", "newborn", "baby sleep", "safe sleep", "sids", "suffocation"],
        "response": [
            "I can hear the love and care in your question about keeping your little one comfortable and safe. Sleep safety is one of those areas where the guidelines have really evolved over the years, and I know it might feel different from what our own parents did.",
            
            "Here's what we know helps keep babies safest while they sleep: the crib is actually best when it's pretty bare - just your baby on a firm mattress with a fitted sheet. I know that might sound stark or even cold, but babies actually sleep really well this way!",
            
            "The reason we skip blankets and pillows for the first year isn't to make things harder - it's because babies' breathing and movement are still developing. Unlike us, they can't easily move their heads if something covers their face, and their little bodies handle temperature differently than ours do.",
            
            "Now, I know you're probably thinking 'but how do I keep my baby warm?' And that's such a natural parent concern! Sleep sacks or wearable blankets are amazing for this - they keep baby cozy without any loose fabric. You can also layer their clothing based on room temperature.",
            
            "A good rule of thumb is to dress baby in one more layer than you'd be comfortable in. Feel their chest (not hands or feet - those are often cooler) to check if they're warm but not sweaty.",
            
            "The bare crib guideline typically lasts until around baby's first birthday, when they're much more mobile and can move around freely. Then you can gradually introduce a small blanket, and pillows usually come later when they transition to a toddler bed.",
            
            "I know these guidelines can feel overwhelming when you just want to create a cozy nest for your little one. Remember, you're doing an amazing job thinking ahead about safety - that's exactly the kind of thoughtful parent your baby needs."
        ]
    },
    "exercise_pregnancy": {
        "keywords": ["exercise", "run", "marathon", "workout", "gym", "fitness", "activity", "pregnant", "pregnancy"],
        "response": [
            "What an exciting goal! I love that you're thinking about staying active - movement during pregnancy can be such a gift to both you and your baby.",
            
            "Marathon running is definitely one of those topics where your individual experience really matters. If you were already running long distances before pregnancy, your body might handle continued training differently than someone just starting out.",
            
            "That said, marathons are pretty intense even for experienced runners, and pregnancy adds some extra considerations. Your body is already working overtime growing a baby, your center of gravity changes, and your joints are more flexible due to hormones.",
            
            "Most healthcare providers lean toward modifying distance and intensity during pregnancy, even for seasoned runners. The general sweet spot tends to be around 150 minutes of moderate activity per week - which could absolutely include running, just maybe shorter distances.",
            
            "What I'd really encourage is having an honest conversation with your doctor or midwife about your running background, current fitness level, and how this pregnancy is going. They know your body and your specific situation best.",
            
            "There are also some wonderful alternatives that might scratch that same itch - like training for a shorter race during pregnancy and saving the marathon goal for after baby arrives. Some people find pregnancy running actually helps them discover new favorite distances!",
            
            "Whatever you decide, listen to your body above all else. If something feels off, it probably is. And remember, there's no 'perfect' way to stay fit during pregnancy - the best exercise is the one that feels good and keeps you both healthy."
        ]
    },
    "breastfeeding": {
        "keywords": ["breastfeed", "nursing", "latch", "milk", "feeding", "breast", "formula", "bottle"],
        "response": [
            "Breastfeeding - there's so much emotion wrapped up in this topic, isn't there? Whether you're just starting to think about it or you're in the thick of it, know that whatever you're feeling is completely normal.",
            
            "If you're just getting started, those first few days can feel like you're both learning a completely new language together. Skin-to-skin time right after birth often helps, but don't worry if it doesn't click immediately - you're both figuring this out.",
            
            "Watch for your baby's early hunger signs - they'll start rooting around, making little sucking motions, or bringing hands to mouth. These cues are usually easier to work with than waiting for crying, which is often a later hunger sign.",
            
            "When it comes to latch, you want baby's mouth to cover a good portion of the darker area around your nipple, not just the tip. It might feel uncomfortable at first, but it shouldn't be consistently painful. If it hurts beyond those first few seconds, it's worth getting some help.",
            
            "Speaking of help - please don't hesitate to reach out if you're struggling. Lactation consultants, your healthcare provider, or even experienced friends can make such a difference. Sometimes just a small positioning adjustment changes everything.",
            
            "And remember, fed is best. Whether that's breastfeeding, formula feeding, or a combination, you're nourishing your baby and that's what matters. There's no prize for suffering through breastfeeding if it's not working for your family.",
            
            "Trust your instincts, be patient with yourself, and know that whatever feeding journey you're on is the right one for you and your baby."
        ]
    },
    "pregnancy_symptoms": {
        "keywords": ["nausea", "morning sickness", "tired", "fatigue", "symptoms", "sick", "vomit", "headache"],
        "response": [
            "Oh, pregnancy symptoms - they can really knock you off your feet, can't they? First off, what you're feeling is so incredibly common, even if it doesn't feel that way when you're in the middle of it.",
            
            "Morning sickness is probably one of the biggest misnomers out there - it can strike any time of day! Around 70-80% of pregnant people experience it, usually starting around 6 weeks. For most, it starts to ease up in the second trimester, though everyone's timeline is different.",
            
            "Fatigue is another big one, especially in early pregnancy. Your body is literally building another human being, plus your blood volume is increasing, hormones are surging - no wonder you feel wiped out! This often gets better in the second trimester too.",
            
            "For nausea, small frequent meals often help more than trying to eat regular-sized meals. Keeping crackers by your bed and nibbling a few before you even get up can sometimes help. Ginger - whether as tea, candy, or even ginger ale - seems to help some people.",
            
            "With fatigue, rest when you can, even if it's just putting your feet up for 10 minutes. Gentle movement like short walks can sometimes help energy levels, but don't push yourself.",
            
            "That said, if you're vomiting so much you can't keep fluids down, losing weight, or having severe abdominal pain, definitely call your healthcare provider. The same goes for heavy bleeding or high fever.",
            
            "Remember, every pregnancy is different. Try not to compare your experience to others - your body is doing exactly what it needs to do for your baby."
        ]
    },
    "postpartum": {
        "keywords": ["postpartum", "after birth", "recovery", "baby blues", "depression", "healing"],
        "response": [
            "The postpartum period - it's such a whirlwind of physical recovery, emotional changes, and learning to care for your new little person. Please be gentle with yourself during this time.",
            
            "Your body has just done something incredible, and healing takes time. For vaginal deliveries, you're looking at about 6-8 weeks for the major healing, longer for C-sections. But honestly, feeling 'back to normal' often takes longer than that, and that's completely okay.",
            
            "Rest really is medicine right now. I know everyone says 'sleep when the baby sleeps,' and I know that can feel impossible with everything else going on. But try to take it seriously - even lying down for 20 minutes can help.",
            
            "Emotionally, those first few weeks can be a rollercoaster. Baby blues are incredibly common - mood swings, crying spells, feeling overwhelmed. This usually peaks around day 3-5 and starts to settle by two weeks.",
            
            "If you're feeling sad, anxious, or disconnected beyond those first two weeks, or if you're having thoughts about harming yourself or baby, please reach out for help. Postpartum depression and anxiety are real, treatable, and nothing to be ashamed of.",
            
            "Accept help when people offer it. Let someone else hold the baby while you shower, or bring you a meal, or throw in a load of laundry. You don't have to do this alone.",
            
            "Be patient with the bonding process too. Not everyone feels that instant overwhelming love - sometimes it grows gradually, and that's normal too.",
            
            "You're learning to be a parent just like your baby is learning to be in the world. Give yourself the same patience and grace you'd give your best friend going through this."
        ]
    }
}

def log_conversation(user_message, bot_response, sources, feedback=None, user_name=None):
    """Log conversations for quality improvement and analysis."""
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "user_name": user_name,
        "user_message": user_message,
        "bot_response": bot_response if isinstance(bot_response, str) else " ".join(bot_response),
        "sources": sources,
        "feedback": feedback,
        "session_id": request.remote_addr
    }
    
    try:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(LOGS_FOLDER, f"conversations_{date_str}.jsonl")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"Error logging conversation: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_built_in_response(query, user_name=None):
    """Check if query matches built-in knowledge and return appropriate response."""
    query_lower = query.lower()
    name_prefix = f"Hi {user_name}! " if user_name else ""
    
    # Check each knowledge topic
    for topic, info in CLEMENTINA_KNOWLEDGE.items():
        # More flexible keyword matching
        matches = 0
        total_keywords = len(info["keywords"])
        
        for keyword in info["keywords"]:
            if keyword in query_lower:
                matches += 1
        
        # If we find any keywords, return this topic (lowered threshold)
        if matches > 0:
            response_parts = info["response"].copy()
            # Add personalized greeting to first paragraph
            if name_prefix:
                response_parts[0] = name_prefix + response_parts[0]
            
            # Add medical disclaimer in conversational tone
            disclaimer = "Of course, I always want to remind you that while I'm here to share information and support, your healthcare provider knows you and your specific situation best. Don't hesitate to reach out to them with any concerns - that's what they're there for!"
            response_parts.append(disclaimer)
            
            print(f"Found match for topic '{topic}' with {matches} keyword matches")
            return response_parts
    
    print(f"No knowledge match found for query: '{query_lower}'")
    return None

def generate_caring_fallback(query, user_name=None):
    """Generate a caring, helpful fallback response."""
    name_prefix = f"{user_name}, " if user_name else ""
    
    return [
        f"Hi {name_prefix}thank you for trusting me with your question about '{query}'. This sounds like such an important topic for you and your family.",
        
        "You know, I want to give you the most helpful and accurate information possible, but I think this particular question might benefit from the expertise of someone who knows your specific situation.",
        
        "Here's what I'd suggest: your healthcare provider, whether that's your doctor, midwife, or pediatrician, would be perfect for this question. They know your medical history and can give you personalized guidance.",
        
        "You might also find it helpful to connect with other parents in your area - sometimes local parenting groups or pregnancy classes can be wonderful sources of support and shared experiences.",
        
        "I'm still here if you have other questions I might be able to help with. Sometimes approaching topics from a different angle can open up new conversations!",
        
        "Remember, asking questions means you're being thoughtful and proactive about your health and your baby's wellbeing. That's exactly the kind of care your family deserves."
    ]

def ask_question(query: str, provider: str = 'local', user_name: str = None):
    """Answers a question using built-in knowledge first, then RAG if available."""
    
    print(f"Processing query: '{query}' for user: {user_name}")
    
    # First, try built-in knowledge
    built_in_response = get_built_in_response(query, user_name)
    if built_in_response:
        print("Found built-in response")
        return built_in_response, ["Built-in Clementina Knowledge"]
    
    print("No built-in response found, using fallback")
    # Fallback response
    return generate_caring_fallback(query, user_name), ["Clementina's General Guidance"]

# Flask Routes
@app.route('/')
def serve_index():
    return render_template('index.html')

@app.route('/set_name', methods=['POST'])
def set_name():
    """Store user's name in session."""
    data = request.json
    name = data.get('name', '').strip()
    if name:
        session['user_name'] = name
        return jsonify({"success": True, "message": f"Nice to meet you, {name}!"})
    return jsonify({"success": False, "message": "Please provide a valid name."})

@app.route('/ask', methods=['POST'])
def handle_ask():
    data = request.json
    query = data.get('message')
    provider = data.get('provider', 'local')
    user_name = session.get('user_name')

    if not query:
        return jsonify({"error": "No message provided."}), 400

    try:
        answer_parts, sources = ask_question(query, provider, user_name)
        
        # Log the conversation
        log_conversation(query, answer_parts, sources, user_name=user_name)
        
        conversation_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(query)) % 10000}"
        
        return jsonify({
            "answer_parts": answer_parts,
            "sources": list(set(sources)),
            "conversation_id": conversation_id
        })
    except Exception as e:
        print(f"Error in handle_ask: {e}")
        fallback_answer = generate_caring_fallback(query, user_name)
        return jsonify({
            "answer_parts": fallback_answer,
            "sources": ["Clementina's Care"],
            "conversation_id": f"fallback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        })

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    data = request.json
    conversation_id = data.get('conversation_id')
    rating = data.get('rating')
    feedback_text = data.get('feedback', '')
    user_name = session.get('user_name')
    
    feedback_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "conversation_id": conversation_id,
        "rating": rating,
        "feedback_text": feedback_text,
        "user_name": user_name,
        "session_id": request.remote_addr
    }
    
    try:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        feedback_file = os.path.join(LOGS_FOLDER, f"feedback_{date_str}.jsonl")
        
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback_entry) + '\n')
        
        return jsonify({"success": "Thank you for your feedback! It helps me improve."})
    except Exception as e:
        print(f"Error logging feedback: {e}")
        return jsonify({"success": "Thank you for your feedback!"})

@app.route('/analytics', methods=['GET'])
def get_analytics():
    """Get basic analytics from conversation logs."""
    try:
        analytics_data = {
            "total_conversations": 0,
            "total_feedback": 0,
            "average_rating": 0,
            "recent_topics": [],
            "user_names": []
        }
        
        # Read recent conversation logs
        import glob
        conversation_files = glob.glob(os.path.join(LOGS_FOLDER, "conversations_*.jsonl"))
        feedback_files = glob.glob(os.path.join(LOGS_FOLDER, "feedback_*.jsonl"))
        
        # Process conversations
        for file_path in conversation_files[-7:]:  # Last 7 days
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            conv = json.loads(line)
                            analytics_data["total_conversations"] += 1
                            if conv.get('user_name'):
                                analytics_data["user_names"].append(conv['user_name'])
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        # Process feedback
        ratings = []
        for file_path in feedback_files[-7:]:  # Last 7 days
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            fb = json.loads(line)
                            analytics_data["total_feedback"] += 1
                            if fb.get('rating'):
                                ratings.append(fb['rating'])
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        if ratings:
            analytics_data["average_rating"] = round(sum(ratings) / len(ratings), 2)
        
        # Unique users
        analytics_data["unique_users"] = len(set(analytics_data["user_names"]))
        analytics_data["user_names"] = list(set(analytics_data["user_names"]))
        
        return jsonify(analytics_data)
        
    except Exception as e:
        print(f"Analytics error: {e}")
        return jsonify({"error": "Could not load analytics"}), 500

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            return jsonify({"success": f"Medical content '{filename}' has been uploaded successfully. RAG integration coming soon!"})
        except Exception as e:
            print(f"Upload error: {e}")
            return jsonify({"error": f"Failed to process file: {e}"}), 500

    return jsonify({"error": "File type not supported."}), 400

if __name__ == '__main__':
    print("🌸 Clementina Health Assistant starting up...")
    print("💬 Conversational tone enabled")
    print("👤 Name collection in main input")
    print("📊 Analytics system active")
    print("📁 Conversation logging enabled")
    app.run(host='0.0.0.0', port=8080, debug=True)