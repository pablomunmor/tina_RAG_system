import json
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import glob

class ConversationAnalytics:
    """Helper class for analyzing conversation logs and feedback."""
    
    def __init__(self, logs_folder='conversation_logs'):
        self.logs_folder = logs_folder
    
    def load_conversations(self, days_back=30):
        """Load conversations from the last N days."""
        conversations = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Find all conversation log files
        pattern = os.path.join(self.logs_folder, "conversations_*.jsonl")
        for file_path in glob.glob(pattern):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            conv = json.loads(line)
                            conv_date = datetime.fromisoformat(conv['timestamp'])
                            if start_date <= conv_date <= end_date:
                                conversations.append(conv)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        return conversations
    
    def load_feedback(self, days_back=30):
        """Load feedback from the last N days."""
        feedback = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        pattern = os.path.join(self.logs_folder, "feedback_*.jsonl")
        for file_path in glob.glob(pattern):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            fb = json.loads(line)
                            fb_date = datetime.fromisoformat(fb['timestamp'])
                            if start_date <= fb_date <= end_date:
                                feedback.append(fb)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        return feedback
    
    def generate_report(self, days_back=7):
        """Generate a comprehensive analytics report."""
        conversations = self.load_conversations(days_back)
        feedback = self.load_feedback(days_back)
        
        # Basic metrics
        total_conversations = len(conversations)
        unique_sessions = len(set(conv['session_id'] for conv in conversations))
        
        # Common topics/keywords
        all_messages = [conv['user_message'].lower() for conv in conversations]
        health_keywords = {
            'pregnancy': ['pregnant', 'pregnancy', 'expecting', 'baby'],
            'breastfeeding': ['breastfeed', 'nursing', 'latch', 'milk'],
            'labor': ['labor', 'contraction', 'delivery', 'birth'],
            'postpartum': ['postpartum', 'after birth', 'recovery'],
            'symptoms': ['symptom', 'pain', 'nausea', 'cramp'],
            'mental_health': ['anxious', 'worried', 'depressed', 'stress']
        }
        
        topic_counts = defaultdict(int)
        for message in all_messages:
            for topic, keywords in health_keywords.items():
                if any(keyword in message for keyword in keywords):
                    topic_counts[topic] += 1
        
        # Feedback analysis
        ratings = [fb['rating'] for fb in feedback if 'rating' in fb]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        rating_distribution = Counter(ratings)
        
        # Recent trends (daily conversation counts)
        daily_counts = defaultdict(int)
        for conv in conversations:
            date = datetime.fromisoformat(conv['timestamp']).date()
            daily_counts[date] += 1
        
        report = {
            'period': f"Last {days_back} days",
            'total_conversations': total_conversations,
            'unique_users': unique_sessions,
            'average_rating': round(avg_rating, 2),
            'total_feedback_responses': len(feedback),
            'top_topics': dict(topic_counts),
            'rating_distribution': dict(rating_distribution),
            'daily_conversation_counts': {str(k): v for k, v in daily_counts.items()},
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def identify_improvement_areas(self):
        """Identify areas where the chatbot could improve based on feedback."""
        feedback = self.load_feedback()
        low_rated = [fb for fb in feedback if fb.get('rating', 5) <= 2]
        
        improvement_areas = []
        for fb in low_rated:
            if fb.get('feedback_text'):
                improvement_areas.append({
                    'rating': fb['rating'],
                    'feedback': fb['feedback_text'],
                    'timestamp': fb['timestamp']
                })
        
        return improvement_areas
    
    def export_report(self, output_file=None):
        """Export analytics report to JSON file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"analytics_report_{timestamp}.json"
        
        report = self.generate_report()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Analytics report exported to {output_file}")
        return output_file

if __name__ == "__main__":
    # Example usage
    analytics = ConversationAnalytics()
    report = analytics.generate_report(days_back=7)
    
    print("=== Clementina Health Analytics Report ===")
    print(f"Period: {report['period']}")
    print(f"Total Conversations: {report['total_conversations']}")
    print(f"Unique Users: {report['unique_users']}")
    print(f"Average Rating: {report['average_rating']}/5")
    print(f"Total Feedback: {report['total_feedback_responses']}")
    print("\nTop Discussion Topics:")
    for topic, count in sorted(report['top_topics'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic.replace('_', ' ').title()}: {count} conversations")
    
    # Check for improvement areas
    improvements = analytics.identify_improvement_areas()
    if improvements:
        print(f"\n⚠️  {len(improvements)} low-rated responses need attention")
    
    # Export full report
    analytics.export_report()