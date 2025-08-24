#!/usr/bin/env python3
"""
Iron Buddy - Advanced AI Gym Coach Bot for Telegram
Extended with SQLite database, exercise videos, step tracking, food logging, and progress graphs
"""

print("🔥 IRON BUDDY STARTING...")

# Imports
try:
    import os
    import json
    import logging
    import sqlite3
    from datetime import datetime, timedelta
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler
    import asyncio
    from typing import Dict, List, Optional, Tuple
    import math
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.dates as mdates
    from collections import defaultdict
    import requests
    import re
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    print("Install with: pip install python-telegram-bot matplotlib requests")
    exit(1)

from dotenv import load_dotenv
import os

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
print(f"✅ Bot token configured for @Ironbuddybot")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Exercise Video Database
EXERCISE_VIDEOS = {
    # Main Compound Lifts
    'squat': 'https://youtu.be/ultWZbUMPL8',
    'back squat': 'https://youtu.be/ultWZbUMPL8',
    'bench press': 'https://youtu.be/rT7DgCr-3pg',
    'bench': 'https://youtu.be/rT7DgCr-3pg',
    'deadlift': 'https://youtu.be/op9kVnSso6Q',
    'overhead press': 'https://youtu.be/2yjwXTZQDDI',
    'ohp': 'https://youtu.be/2yjwXTZQDDI',
    'military press': 'https://youtu.be/2yjwXTZQDDI',
    
    # Upper Body
    'pull ups': 'https://youtu.be/eGo4IYlbE5g',
    'pullups': 'https://youtu.be/eGo4IYlbE5g',
    'chin ups': 'https://youtu.be/eGo4IYlbE5g',
    'barbell rows': 'https://youtu.be/FWJR5Ve8bnQ',
    'bent over rows': 'https://youtu.be/FWJR5Ve8bnQ',
    'incline bench': 'https://youtu.be/jjLdvoJXki0',
    'incline press': 'https://youtu.be/jjLdvoJXki0',
    'dips': 'https://youtu.be/2z8JmcrW-As',
    'lateral raises': 'https://youtu.be/3VcKaXpzqRo',
    
    # Lower Body  
    'front squat': 'https://youtu.be/uYumuL_G_V0',
    'romanian deadlift': 'https://youtu.be/cn_7m94K6d8',
    'rdl': 'https://youtu.be/cn_7m94K6d8',
    'bulgarian split squats': 'https://youtu.be/2C-uNgKwPLE',
    'lunges': 'https://youtu.be/QE_pHp8dHnY',
    'leg press': 'https://youtu.be/yb1KyjdElzA',
    'leg curls': 'https://youtu.be/ELOCsoDSmrg',
    'calf raises': 'https://youtu.be/gwWv7aPcD24',
    
    # Arms
    'barbell curls': 'https://youtu.be/ykJmrZ5v0Oo',
    'bicep curls': 'https://youtu.be/ykJmrZ5v0Oo',
    'hammer curls': 'https://youtu.be/zC3nLlEvin4',
    'tricep extensions': 'https://youtu.be/2-LAMcpzODU',
    'close grip bench': 'https://youtu.be/nEF0bv2FW94',
    'tricep dips': 'https://youtu.be/2z8JmcrW-As'
}

class DatabaseManager:
    def __init__(self, db_path='iron_buddy.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create all necessary tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                weight REAL,
                height REAL,
                goal TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Workout history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                exercise_name TEXT,
                weight REAL,
                reps INTEGER,
                rpe INTEGER,
                estimated_1rm REAL,
                workout_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Daily steps table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                steps INTEGER,
                step_date DATE,
                logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Food logging table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS food_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                food_description TEXT,
                calories REAL,
                protein REAL,
                carbs REAL,
                fat REAL,
                meal_date DATE,
                logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        # Weight tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weight_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                weight REAL,
                log_date DATE,
                logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    
    def get_user_profile(self, user_id: int) -> Dict:
        """Get or create user profile"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
        user = cursor.fetchone()
        
        if not user:
            # Create new user
            cursor.execute('''
                INSERT INTO users (user_id, username, first_name, weight, height, goal)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, '', '', 0, 0, 'bulk'))
            conn.commit()
            
            # Fetch the newly created user
            cursor.execute('SELECT * FROM users WHERE user_id = ?', (user_id,))
            user = cursor.fetchone()
        
        conn.close()
        
        return {
            'user_id': user[0],
            'username': user[1],
            'first_name': user[2],
            'weight': user[3],
            'height': user[4],
            'goal': user[5],
            'created_at': user[6]
        }
    
    def update_user_stats(self, user_id: int, weight: float = None, height: float = None, goal: str = None):
        """Update user statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = []
        params = []
        
        if weight is not None:
            updates.append('weight = ?')
            params.append(weight)
        if height is not None:
            updates.append('height = ?')
            params.append(height)
        if goal is not None:
            updates.append('goal = ?')
            params.append(goal)
        
        if updates:
            params.append(user_id)
            query = f'UPDATE users SET {", ".join(updates)} WHERE user_id = ?'
            cursor.execute(query, params)
            conn.commit()
        
        conn.close()
    
    def log_workout(self, user_id: int, exercise_name: str, weight: float, reps: int, rpe: int, estimated_1rm: float):
        """Log a workout"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO workouts (user_id, exercise_name, weight, reps, rpe, estimated_1rm)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, exercise_name, weight, reps, rpe, estimated_1rm))
        
        conn.commit()
        conn.close()
    
    def get_workout_history(self, user_id: int, exercise_name: str = None, limit: int = 50) -> List[Dict]:
        """Get workout history for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if exercise_name:
            cursor.execute('''
                SELECT * FROM workouts 
                WHERE user_id = ? AND exercise_name = ? 
                ORDER BY workout_date DESC 
                LIMIT ?
            ''', (user_id, exercise_name, limit))
        else:
            cursor.execute('''
                SELECT * FROM workouts 
                WHERE user_id = ? 
                ORDER BY workout_date DESC 
                LIMIT ?
            ''', (user_id, limit))
        
        workouts = cursor.fetchall()
        conn.close()
        
        return [{
            'id': w[0],
            'user_id': w[1],
            'exercise_name': w[2],
            'weight': w[3],
            'reps': w[4],
            'rpe': w[5],
            'estimated_1rm': w[6],
            'workout_date': w[7]
        } for w in workouts]
    
    def log_steps(self, user_id: int, steps: int, step_date: str = None):
        """Log daily steps"""
        if not step_date:
            step_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if entry exists for today
        cursor.execute('SELECT id FROM daily_steps WHERE user_id = ? AND step_date = ?', (user_id, step_date))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing entry
            cursor.execute('UPDATE daily_steps SET steps = ? WHERE id = ?', (steps, existing[0]))
        else:
            # Create new entry
            cursor.execute('INSERT INTO daily_steps (user_id, steps, step_date) VALUES (?, ?, ?)',
                          (user_id, steps, step_date))
        
        conn.commit()
        conn.close()
    
    def get_steps_history(self, user_id: int, days: int = 7) -> List[Dict]:
        """Get recent steps history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM daily_steps 
            WHERE user_id = ? 
            ORDER BY step_date DESC 
            LIMIT ?
        ''', (user_id, days))
        
        steps = cursor.fetchall()
        conn.close()
        
        return [{
            'id': s[0],
            'user_id': s[1],
            'steps': s[2],
            'step_date': s[3],
            'logged_at': s[4]
        } for s in steps]
    
    def log_food(self, user_id: int, food_description: str, calories: float, protein: float, carbs: float, fat: float):
        """Log food intake"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        meal_date = datetime.now().strftime('%Y-%m-%d')
        
        cursor.execute('''
            INSERT INTO food_logs (user_id, food_description, calories, protein, carbs, fat, meal_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, food_description, calories, protein, carbs, fat, meal_date))
        
        conn.commit()
        conn.close()
    
    def get_daily_nutrition(self, user_id: int, date: str = None) -> Dict:
        """Get nutrition totals for a specific day"""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT SUM(calories), SUM(protein), SUM(carbs), SUM(fat)
            FROM food_logs 
            WHERE user_id = ? AND meal_date = ?
        ''', (user_id, date))
        
        totals = cursor.fetchone()
        conn.close()
        
        return {
            'calories': totals[0] or 0,
            'protein': totals[1] or 0,
            'carbs': totals[2] or 0,
            'fat': totals[3] or 0
        }
    
    def get_weekly_nutrition(self, user_id: int) -> List[Dict]:
        """Get nutrition for the past 7 days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT meal_date, SUM(calories), SUM(protein), SUM(carbs), SUM(fat)
            FROM food_logs 
            WHERE user_id = ? AND meal_date >= ?
            GROUP BY meal_date
            ORDER BY meal_date DESC
        ''', (user_id, week_ago))
        
        weekly_data = cursor.fetchall()
        conn.close()
        
        return [{
            'date': w[0],
            'calories': w[1] or 0,
            'protein': w[2] or 0,
            'carbs': w[3] or 0,
            'fat': w[4] or 0
        } for w in weekly_data]
    
    def log_weight(self, user_id: int, weight: float):
        """Log body weight"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        log_date = datetime.now().strftime('%Y-%m-%d')
        
        cursor.execute('INSERT INTO weight_logs (user_id, weight, log_date) VALUES (?, ?, ?)',
                      (user_id, weight, log_date))
        
        conn.commit()
        conn.close()
        
        # Also update user profile
        self.update_user_stats(user_id, weight=weight)
    
    def get_weight_history(self, user_id: int, days: int = 30) -> List[Dict]:
        """Get weight history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM weight_logs 
            WHERE user_id = ? 
            ORDER BY log_date DESC 
            LIMIT ?
        ''', (user_id, days))
        
        weights = cursor.fetchall()
        conn.close()
        
        return [{
            'id': w[0],
            'user_id': w[1],
            'weight': w[2],
            'log_date': w[3],
            'logged_at': w[4]
        } for w in weights]

class NutritionEstimator:
    """Simple nutrition estimator using pattern matching"""
    
    def __init__(self):
        # Basic nutrition database (per 100g)
        self.nutrition_db = {
            # Proteins
            'chicken': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6},
            'chicken breast': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6},
            'egg': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11},  # per egg (50g)
            'eggs': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11},
            'fish': {'calories': 200, 'protein': 25, 'carbs': 0, 'fat': 10},
            'beef': {'calories': 250, 'protein': 26, 'carbs': 0, 'fat': 17},
            'milk': {'calories': 60, 'protein': 3.2, 'carbs': 4.8, 'fat': 3.2},  # per 100ml
            'paneer': {'calories': 265, 'protein': 18, 'carbs': 1.2, 'fat': 20},
            'dal': {'calories': 116, 'protein': 9, 'carbs': 20, 'fat': 0.4},
            'lentils': {'calories': 116, 'protein': 9, 'carbs': 20, 'fat': 0.4},
            
            # Carbs
            'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3},
            'roti': {'calories': 120, 'protein': 3, 'carbs': 18, 'fat': 3},  # per roti (30g)
            'chapati': {'calories': 120, 'protein': 3, 'carbs': 18, 'fat': 3},
            'bread': {'calories': 80, 'protein': 2.5, 'carbs': 15, 'fat': 1},  # per slice
            'oats': {'calories': 389, 'protein': 17, 'carbs': 66, 'fat': 7},
            'banana': {'calories': 89, 'protein': 1.1, 'carbs': 23, 'fat': 0.3},  # per banana
            'apple': {'calories': 52, 'protein': 0.3, 'carbs': 14, 'fat': 0.2},  # per apple
            'potato': {'calories': 77, 'protein': 2, 'carbs': 17, 'fat': 0.1},
            
            # Fats
            'oil': {'calories': 884, 'protein': 0, 'carbs': 0, 'fat': 100},  # per tbsp (14g)
            'ghee': {'calories': 900, 'protein': 0, 'carbs': 0, 'fat': 100},
            'nuts': {'calories': 600, 'protein': 20, 'carbs': 20, 'fat': 50},
            'almonds': {'calories': 579, 'protein': 21, 'carbs': 22, 'fat': 50}
        }
    
    def estimate_nutrition(self, food_description: str) -> Dict:
        """Estimate nutrition from food description"""
        food_description = food_description.lower().strip()
        
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0
        
        # Extract quantities and foods using regex
        # Pattern: number + unit + food
        patterns = [
            r'(\d+(?:\.\d+)?)\s*(kg|grams?|g)\s+([a-zA-Z\s]+)',  # 100g chicken
            r'(\d+)\s+(cups?|cup)\s+([a-zA-Z\s]+)',  # 1 cup rice
            r'(\d+)\s+(tbsp|tablespoons?)\s+([a-zA-Z\s]+)',  # 2 tbsp oil
            r'(\d+)\s+([a-zA-Z\s]+)',  # 2 eggs, 3 roti
        ]
        
        found_items = []
        
        for pattern in patterns:
            matches = re.findall(pattern, food_description)
            for match in matches:
                if len(match) == 3:
                    quantity, unit, food_name = match
                    quantity = float(quantity)
                    food_name = food_name.strip()
                    found_items.append((quantity, unit, food_name))
        
        # If no structured format found, try to identify foods directly
        if not found_items:
            for food_name in self.nutrition_db.keys():
                if food_name in food_description:
                    found_items.append((1, 'serving', food_name))
        
        for quantity, unit, food_name in found_items:
            # Find best matching food
            best_match = None
            for food_key in self.nutrition_db.keys():
                if food_key in food_name.lower():
                    best_match = food_key
                    break
            
            if best_match:
                nutrition = self.nutrition_db[best_match]
                
                # Convert units to standard portions
                multiplier = quantity
                
                if unit in ['g', 'grams', 'gram']:
                    if best_match in ['egg', 'eggs']:
                        multiplier = quantity / 50  # 1 egg = 50g
                    elif best_match in ['roti', 'chapati']:
                        multiplier = quantity / 30  # 1 roti = 30g
                    elif best_match in ['bread']:
                        multiplier = quantity / 25  # 1 slice = 25g
                    else:
                        multiplier = quantity / 100  # per 100g
                
                elif unit in ['kg']:
                    multiplier = quantity * 10  # 1kg = 1000g = 10 servings of 100g
                
                elif unit in ['cup', 'cups']:
                    if best_match == 'rice':
                        multiplier = quantity * 200 / 100  # 1 cup cooked rice = 200g
                    elif best_match == 'milk':
                        multiplier = quantity * 240 / 100  # 1 cup milk = 240ml
                    else:
                        multiplier = quantity * 1.5  # approximate
                
                elif unit in ['tbsp', 'tablespoons']:
                    multiplier = quantity * 14 / 100  # 1 tbsp = 14g
                
                # Add to totals
                total_calories += nutrition['calories'] * multiplier
                total_protein += nutrition['protein'] * multiplier
                total_carbs += nutrition['carbs'] * multiplier
                total_fat += nutrition['fat'] * multiplier
        
        # If nothing found, provide rough estimates
        if total_calories == 0:
            total_calories = 400  # Default meal estimate
            total_protein = 25
            total_carbs = 45
            total_fat = 15
        
        return {
            'calories': round(total_calories, 1),
            'protein': round(total_protein, 1),
            'carbs': round(total_carbs, 1),
            'fat': round(total_fat, 1)
        }

class GraphGenerator:
    """Generate progress graphs using matplotlib"""
    
    @staticmethod
    def create_weight_progress_graph(weight_data: List[Dict], user_name: str) -> str:
        """Create weight progress graph"""
        if not weight_data:
            return None
        
        dates = [datetime.strptime(w['log_date'], '%Y-%m-%d') for w in reversed(weight_data)]
        weights = [w['weight'] for w in reversed(weight_data)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(dates, weights, marker='o', linewidth=2, markersize=8, color='#FF6B6B')
        
        plt.title(f'🏋️ {user_name}\'s Weight Progress', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Weight (kg)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//7)))
        plt.xticks(rotation=45)
        
        # Add trend line
        if len(weights) > 1:
            z = np.polyfit(range(len(weights)), weights, 1)
            p = np.poly1d(z)
            plt.plot(dates, p(range(len(weights))), "--", color='gray', alpha=0.8)
            
            # Calculate change
            weight_change = weights[-1] - weights[0]
            days_elapsed = (dates[-1] - dates[0]).days
            
            if days_elapsed > 0:
                weekly_change = (weight_change / days_elapsed) * 7
                color = 'green' if weight_change < 0 else 'red'
                plt.text(0.02, 0.98, f'Change: {weight_change:+.1f}kg\n({weekly_change:+.1f}kg/week)',
                        transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.1))
        
        plt.tight_layout()
        
        # Save and return filename
        filename = f'weight_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename
    
    @staticmethod
    def create_exercise_progress_graph(workout_data: List[Dict], exercise_name: str, user_name: str) -> str:
        """Create exercise progress graph"""
        if not workout_data:
            return None
        
        dates = [datetime.strptime(w['workout_date'].split()[0], '%Y-%m-%d') for w in reversed(workout_data)]
        estimated_1rms = [w['estimated_1rm'] for w in reversed(workout_data)]
        
        plt.figure(figsize=(12, 8))
        
        # Main progress line
        plt.subplot(2, 1, 1)
        plt.plot(dates, estimated_1rms, marker='o', linewidth=3, markersize=8, color='#4ECDC4')
        plt.title(f'💪 {user_name}\'s {exercise_name.title()} Progress (Estimated 1RM)', fontsize=16, fontweight='bold')
        plt.ylabel('Estimated 1RM (kg)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
        plt.xticks(rotation=45)
        
        # Add trend line and stats
        if len(estimated_1rms) > 1:
            z = np.polyfit(range(len(estimated_1rms)), estimated_1rms, 1)
            p = np.poly1d(z)
            plt.plot(dates, p(range(len(estimated_1rms))), "--", color='orange', alpha=0.8, linewidth=2)
            
            # Calculate progress
            progress = estimated_1rms[-1] - estimated_1rms[0]
            progress_percent = (progress / estimated_1rms[0]) * 100 if estimated_1rms[0] > 0 else 0
            
            plt.text(0.02, 0.98, f'Progress: {progress:+.1f}kg ({progress_percent:+.1f}%)\nSessions: {len(workout_data)}',
                    transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        # Volume chart (weight x reps)
        plt.subplot(2, 1, 2)
        volumes = [w['weight'] * w['reps'] for w in reversed(workout_data)]
        plt.bar(range(len(volumes)), volumes, alpha=0.7, color='#FF9F43', width=0.8)
        plt.title('📊 Training Volume (Weight × Reps)', fontsize=14, fontweight='bold')
        plt.ylabel('Volume (kg)', fontsize=12)
        plt.xlabel('Session Number', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save and return filename
        filename = f'{exercise_name.replace(" ", "_")}_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filename

# Initialize components
db = DatabaseManager()
nutrition_estimator = NutritionEstimator()
print("✅ Database and components initialized")

def calculate_1rm(weight: float, reps: int, rpe: int) -> float:
    """Calculate estimated 1RM using RPE"""
    rpe_multipliers = {10: 1.0, 9: 1.03, 8: 1.07, 7: 1.11, 6: 1.15, 5: 1.20}
    multiplier = rpe_multipliers.get(rpe, 1.0)
    estimated_1rm = weight * (1 + reps/30) * multiplier
    return round(estimated_1rm, 1)

# ==================== BOT COMMANDS ====================

async def start(update: Update, context):
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    
    # Get or create user profile
    profile = db.get_user_profile(user_id)
    
    welcome_msg = f"""🔥 **WELCOME TO IRON BUDDY ADVANCED** 🔥

What's up {user_name}! I'm your AI powerlifting & bodybuilding coach! 💪

**FEATURES:**
📈 Your 1 Rep max calculated automatically
📊 SQLite Database - Your data is saved forever!
🎥 Exercise Videos - Get form tutorials
👟 Step Tracking - Log your daily steps  
🍽️ Smart Food Logging - Track your nutrition
📈 Progress Graphs - Visual progress tracking

**Quick Start:**
1. `/profile` - Set up your profile
2. `/workout push` - Get a workout
3. `/log bench press 100 5 8` - Log sets
4. `/video squat` - Get exercise videos
5. `/food 2 eggs 1 roti chicken` - Log meals

Type `/help` for all commands!

Ready to get STRONG? 🔥💪"""
    
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context):
    help_text = """🔥 **IRON BUDDY ADVANCED COMMANDS** 🔥

**👤 PROFILE & SETUP:**
/profile - View/update your profile
/stats [weight] [goal] - Quick stats update

**🏋️ WORKOUT TRACKING:**
/log [exercise] [weight] [reps] [rpe] - Log workout sets
/workout [push/pull/legs] - Get workout plans
/history [exercise] - View workout history
/progress [exercise] - Exercise progress graphs

**🎥 EXERCISE HELP:**
/video [exercise] - Get exercise tutorial videos
/videos - List all available exercise videos

**🍽️ NUTRITION TRACKING:**
/food [description] - Log meals (e.g., "2 eggs 1 roti chicken")
/nutrition - View today's nutrition
/nutrition_week - View weekly nutrition summary

**👟 FITNESS TRACKING:**
/steps [number] - Log daily steps
/steps_week - View weekly steps

**📈 PROGRESS & ANALYTICS:**
/weight [kg] - Log body weight
/weight_graph - Generate weight progress graph
/1rm [weight] [reps] [rpe] - Calculate 1RM

**Examples:**
• `/log squat 100 5 8`
• `/video bench press`
• `/food 100g chicken 1 cup rice 2 tbsp oil`
• `/steps 8500`
• `/weight 75`

Let's get STRONG! 💪🔥"""
    
    await update.message.reply_text(help_text)

async def profile_command(update: Update, context):
    user_id = update.effective_user.id
    profile = db.get_user_profile(user_id)
    
    if not context.args:
        # Display current profile
        response = f"""👤 **YOUR PROFILE**

**Basic Info:**
• Name: {update.effective_user.first_name}
• Weight: {profile['weight']}kg
• Height: {profile['height']}cm
• Goal: {profile['goal'].title()}

**Update Examples:**
• `/profile weight 75` - Update weight
• `/profile height 175` - Update height  
• `/profile goal cut` - Update goal
• `/profile 75 175 bulk` - Update all

**Stats:**
• Member since: {profile['created_at'][:10]}"""
        
        await update.message.reply_text(response)
        return
    
    try:
        if len(context.args) == 2:
            # Update single field
            field, value = context.args[0].lower(), context.args[1]
            
            if field == 'weight':
                db.update_user_stats(user_id, weight=float(value))
                await update.message.reply_text(f"✅ Weight updated to {value}kg!")
                
            elif field == 'height':
                db.update_user_stats(user_id, height=float(value))
                await update.message.reply_text(f"✅ Height updated to {value}cm!")
                
            elif field == 'goal':
                db.update_user_stats(user_id, goal=value.lower())
                await update.message.reply_text(f"✅ Goal updated to {value}!")
                
        elif len(context.args) == 3:
            # Update all: weight, height, goal
            weight, height, goal = float(context.args[0]), float(context.args[1]), context.args[2]
            db.update_user_stats(user_id, weight=weight, height=height, goal=goal.lower())
            
            response = f"""✅ **PROFILE UPDATED!**

• Weight: {weight}kg
• Height: {height}cm  
• Goal: {goal.title()}

Ready to crush your goals! 💪"""
            
            await update.message.reply_text(response)
            
    except ValueError:
        await update.message.reply_text("❌ Use: `/profile weight 75` or `/profile 75 175 bulk`")

async def log_workout_command(update: Update, context):
    user_id = update.effective_user.id
    
    if len(context.args) < 4:
        await update.message.reply_text("""❌ **Format:** `/log [exercise] [weight] [reps] [rpe]`

**Examples:**
• `/log bench press 100 5 8`
• `/log squat 140 3 9`""")
        return
    
    try:
        # Parse input - find where numbers start
        exercise_parts = []
        weight = reps = rpe = None
        
        for i, arg in enumerate(context.args):
            try:
                weight = float(arg)
                exercise_parts = context.args[:i]
                reps = int(context.args[i+1])
                rpe = int(context.args[i+2])
                break
            except (ValueError, IndexError):
                continue
        
        if not all([exercise_parts, weight, reps, rpe]):
            raise ValueError("Could not parse")
        
        exercise_name = " ".join(exercise_parts).title()
        
        if not 1 <= rpe <= 10:
            await update.message.reply_text("❌ RPE must be 1-10!")
            return
        
        # Calculate 1RM and save to database
        estimated_1rm = calculate_1rm(weight, reps, rpe)
        db.log_workout(user_id, exercise_name, weight, reps, rpe, estimated_1rm)
        
        # Generate response with advice
        if rpe <= 7:
            advice = f"Felt easy @{rpe}! Try {weight * 1.025:.1f}kg next time 💪"
        elif rpe == 8:
            advice = f"Perfect @8! Try {weight * 1.0125:.1f}kg next session"
        else:
            advice = f"That was tough @{rpe}! Same weight next time, aim for @8"
        
        response = f"""✅ **WORKOUT LOGGED TO DATABASE!**

**{exercise_name}**: {weight}kg x {reps} @{rpe} RPE
**Est. 1RM**: {estimated_1rm}kg

🎯 **Next time**: {advice}

📊 View progress: `/progress {exercise_name.lower()}`
🎥 Need form help? `/video {exercise_name.lower()}`

Great work! 🔥"""
        
        await update.message.reply_text(response)
        
    except Exception as e:
        await update.message.reply_text("❌ Error! Use: `/log bench press 100 5 8`")

async def workout_history_command(update: Update, context):
    user_id = update.effective_user.id
    
    if context.args:
        # Specific exercise history
        exercise_name = " ".join(context.args).title()
        workouts = db.get_workout_history(user_id, exercise_name, limit=10)
        
        if not workouts:
            await update.message.reply_text(f"❌ No history found for {exercise_name}")
            return
        
        response = f"📊 **{exercise_name.upper()} HISTORY**\n\n"
        
        for workout in workouts:
            date = workout['workout_date'][:10]  # YYYY-MM-DD
            response += f"• {date}: {workout['weight']}kg x {workout['reps']} @{workout['rpe']} (1RM: {workout['estimated_1rm']}kg)\n"
        
        response += f"\n📈 View progress graph: `/progress {exercise_name.lower()}`"
        
    else:
        # All recent workouts
        workouts = db.get_workout_history(user_id, limit=15)
        
        if not workouts:
            await update.message.reply_text("❌ No workout history found! Start logging with `/log`")
            return
        
        response = "📊 **RECENT WORKOUT HISTORY**\n\n"
        
        for workout in workouts:
            date = workout['workout_date'][:10]
            response += f"• {date}: {workout['exercise_name']} - {workout['weight']}kg x {workout['reps']} @{workout['rpe']}\n"
        
        response += f"\n📝 Total logged workouts: {len(workouts)}"
    
    await update.message.reply_text(response)

async def progress_graph_command(update: Update, context):
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    
    if not context.args:
        await update.message.reply_text("""📈 **PROGRESS GRAPHS**

**Examples:**
• `/progress bench press` - Bench progress
• `/progress squat` - Squat progress
• `/progress deadlift` - Deadlift progress

Shows estimated 1RM progress and training volume over time!""")
        return
    
    exercise_name = " ".join(context.args).title()
    workouts = db.get_workout_history(user_id, exercise_name, limit=50)
    
    if len(workouts) < 2:
        await update.message.reply_text(f"❌ Need at least 2 {exercise_name} sessions for progress graph!")
        return
    
    try:
        # Generate graph
        filename = GraphGenerator.create_exercise_progress_graph(workouts, exercise_name, user_name)
        
        if filename:
            with open(filename, 'rb') as graph_file:
                await update.message.reply_photo(
                    photo=graph_file,
                    caption=f"📈 **{exercise_name.upper()} PROGRESS** 💪\n\nYour gains visualized! Keep crushing it! 🔥"
                )
            
            # Clean up file
            os.remove(filename)
        else:
            await update.message.reply_text("❌ Error generating progress graph")
            
    except Exception as e:
        await update.message.reply_text(f"❌ Error creating graph: {str(e)}")



async def exercise_video_command(update: Update, context):
    if not context.args:
        await update.message.reply_text("""🎥 **EXERCISE VIDEOS**

**Format:** `/video [exercise name]`

**Examples:**
• `/video squat`
• `/video bench press`
• `/video deadlift`

**See all videos:** `/videos`""")
        return
    
    exercise_name = " ".join(context.args).lower().strip()
    
    # Find matching video
    video_url = None
    matched_exercise = None
    
    # Exact match first
    if exercise_name in EXERCISE_VIDEOS:
        video_url = EXERCISE_VIDEOS[exercise_name]
        matched_exercise = exercise_name
    else:
        # Partial match
        for exercise_key in EXERCISE_VIDEOS.keys():
            if exercise_name in exercise_key or exercise_key in exercise_name:
                video_url = EXERCISE_VIDEOS[exercise_key]
                matched_exercise = exercise_key
                break
    
    if video_url:
        response = f"""🎥 **{matched_exercise.upper()} TUTORIAL**

**Video Link:** {video_url}

**Key Points:**
• Watch full video before attempting
• Start with light weight to practice form
• Focus on controlled movement
• Ask a trainer if unsure!

💪 Perfect your form, then add weight! 🔥"""
        
        await update.message.reply_text(response)
    else:
        await update.message.reply_text(f"❌ No video found for '{exercise_name}'\n\nTry `/videos` to see available exercises!")

async def list_videos_command(update: Update, context):
    response = """🎥 **AVAILABLE EXERCISE VIDEOS**

**🏋️ MAIN LIFTS:**
• squat, bench press, deadlift
• overhead press, front squat

**💪 UPPER BODY:**
• pull ups, barbell rows, incline bench
• dips, lateral raises, bicep curls
• hammer curls, tricep extensions

**🦵 LOWER BODY:**
• romanian deadlift, bulgarian split squats
• lunges, leg press, leg curls, calf raises

**Usage:** `/video [exercise name]`
**Example:** `/video bench press`

All videos are free YouTube tutorials! 🎯"""
    
    await update.message.reply_text(response)

async def food_logging_command(update: Update, context):
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text("""🍽️ **FOOD LOGGING**

**Format:** `/food [food description]`

**Examples:**
• `/food 100g chicken 1 cup rice 2 tbsp oil`
• `/food 2 eggs 1 roti banana`
• `/food 200g paneer dal roti ghee`

**Supported foods:** chicken, eggs, rice, roti, dal, paneer, milk, oats, banana, apple, and more!

**View nutrition:** `/nutrition`""")
        return
    
    food_description = " ".join(context.args)
    
    # Estimate nutrition
    nutrition = nutrition_estimator.estimate_nutrition(food_description)
    
    # Log to database
    db.log_food(user_id, food_description, 
                nutrition['calories'], nutrition['protein'], 
                nutrition['carbs'], nutrition['fat'])
    
    response = f"""✅ **FOOD LOGGED!**

**Meal:** {food_description}

**Estimated Nutrition:**
• 🔥 Calories: {nutrition['calories']:.0f}
• 🥩 Protein: {nutrition['protein']:.1f}g
• 🍞 Carbs: {nutrition['carbs']:.1f}g  
• 🥑 Fat: {nutrition['fat']:.1f}g

📊 **View daily totals:** `/nutrition`
📈 **Weekly summary:** `/nutrition_week`

Great fuel for your gains! 💪"""
    
    await update.message.reply_text(response)

async def daily_nutrition_command(update: Update, context):
    user_id = update.effective_user.id
    profile = db.get_user_profile(user_id)
    
    # Get today's nutrition
    daily_nutrition = db.get_daily_nutrition(user_id)
    
    if daily_nutrition['calories'] == 0:
        await update.message.reply_text("🍽️ No food logged today! Start with `/food [description]`")
        return
    
    # Calculate targets based on profile
    weight = profile['weight'] if profile['weight'] > 0 else 70
    goal = profile['goal']
    
    if 'bulk' in goal:
        target_calories = weight * 35
        target_protein = weight * 2.2
    elif 'cut' in goal:
        target_calories = weight * 28
        target_protein = weight * 2.5
    else:
        target_calories = weight * 32
        target_protein = weight * 2.0
    
    # Calculate percentages
    cal_percent = (daily_nutrition['calories'] / target_calories) * 100
    protein_percent = (daily_nutrition['protein'] / target_protein) * 100
    
    response = f"""📊 **TODAY'S NUTRITION** ({goal.upper()})

**Current Intake:**
• 🔥 Calories: {daily_nutrition['calories']:.0f} / {target_calories:.0f} ({cal_percent:.0f}%)
• 🥩 Protein: {daily_nutrition['protein']:.1f}g / {target_protein:.0f}g ({protein_percent:.0f}%)
• 🍞 Carbs: {daily_nutrition['carbs']:.1f}g
• 🥑 Fat: {daily_nutrition['fat']:.1f}g

**Progress Bars:**
Calories: {'█' * int(cal_percent/10)}{'░' * (10-int(cal_percent/10))}
Protein: {'█' * int(protein_percent/10)}{'░' * (10-int(protein_percent/10))}

**Remaining:**
• Calories: {target_calories - daily_nutrition['calories']:.0f}
• Protein: {target_protein - daily_nutrition['protein']:.1f}g

🍽️ **Add more:** `/food [description]`"""
    
    await update.message.reply_text(response)

async def weekly_nutrition_command(update: Update, context):
    user_id = update.effective_user.id
    weekly_data = db.get_weekly_nutrition(user_id)
    
    if not weekly_data:
        await update.message.reply_text("📊 No nutrition data for this week! Start logging with `/food`")
        return
    
    response = "📈 **WEEKLY NUTRITION SUMMARY**\n\n"
    
    total_calories = 0
    total_protein = 0
    
    for day_data in weekly_data:
        date = day_data['date']
        calories = day_data['calories']
        protein = day_data['protein']
        
        total_calories += calories
        total_protein += protein
        
        response += f"**{date}:** {calories:.0f} cal, {protein:.0f}g protein\n"
    
    avg_calories = total_calories / len(weekly_data)
    avg_protein = total_protein / len(weekly_data)
    
    response += f"""
**📊 WEEKLY TOTALS:**
• Total Calories: {total_calories:.0f}
• Total Protein: {total_protein:.0f}g
• Days Logged: {len(weekly_data)}

**📊 DAILY AVERAGES:**
• Calories: {avg_calories:.0f}/day
• Protein: {avg_protein:.0f}g/day

Keep up the consistency! 💪"""
    
    await update.message.reply_text(response)

async def steps_logging_command(update: Update, context):
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text("""👟 **STEP TRACKING**

**Format:** `/steps [number]`

**Examples:**
• `/steps 8500` - Log today's steps
• `/steps 12000` - Log 12k steps

**View weekly:** `/steps_week`

Target: 8,000-10,000 steps/day! 🚶‍♂️""")
        return
    
    try:
        steps = int(context.args[0])
        
        if steps < 0 or steps > 50000:
            await update.message.reply_text("❌ Steps must be between 0-50,000!")
            return
        
        # Log steps to database
        db.log_steps(user_id, steps)
        
        # Generate motivational response
        if steps >= 12000:
            motivation = "🔥 BEAST MODE! Amazing step count!"
        elif steps >= 10000:
            motivation = "🎯 Perfect! Hit that 10k target!"
        elif steps >= 8000:
            motivation = "💪 Great job! Solid daily activity!"
        elif steps >= 5000:
            motivation = "👍 Good start! Try for 8k+ tomorrow!"
        else:
            motivation = "📈 Every step counts! Keep building!"
        
        response = f"""✅ **STEPS LOGGED!**

**Today's Steps:** {steps:,}
**Status:** {motivation}

**Daily Targets:**
• 🥉 Basic: 5,000 steps
• 🥈 Good: 8,000 steps  
• 🥇 Excellent: 10,000+ steps

📊 **View weekly:** `/steps_week`

Keep moving! 🚶‍♂️💨"""
        
        await update.message.reply_text(response)
        
    except ValueError:
        await update.message.reply_text("❌ Use numbers only! Example: `/steps 8500`")

async def weekly_steps_command(update: Update, context):
    user_id = update.effective_user.id
    steps_data = db.get_steps_history(user_id, days=7)
    
    if not steps_data:
        await update.message.reply_text("👟 No steps logged yet! Start with `/steps [number]`")
        return
    
    response = "📊 **WEEKLY STEPS SUMMARY**\n\n"
    
    total_steps = 0
    for day in steps_data:
        date = day['step_date']
        steps = day['steps']
        total_steps += steps
        
        # Add status emoji
        if steps >= 10000:
            status = "🥇"
        elif steps >= 8000:
            status = "🥈"
        elif steps >= 5000:
            status = "🥉"
        else:
            status = "📈"
        
        response += f"{status} **{date}:** {steps:,} steps\n"
    
    avg_steps = total_steps / len(steps_data)
    
    response += f"""
**📊 WEEKLY TOTALS:**
• Total Steps: {total_steps:,}
• Daily Average: {avg_steps:,.0f}
• Days Logged: {len(steps_data)}

**🎯 STATUS:** """
    
    if avg_steps >= 10000:
        response += "EXCELLENT! 🔥"
    elif avg_steps >= 8000:
        response += "GREAT! 💪"
    elif avg_steps >= 5000:
        response += "GOOD PROGRESS! 👍"
    else:
        response += "KEEP BUILDING! 📈"
    
    response += "\n\n👟 Stay active! Every step counts!"
    
    await update.message.reply_text(response)

async def weight_logging_command(update: Update, context):
    user_id = update.effective_user.id
    
    if not context.args:
        await update.message.reply_text("""     **WEIGHT TRACKING**

**Format:** `/weight [kg]`

**Examples:**
• `/weight 75` - Log 75kg
• `/weight 68.5` - Log 68.5kg

**View graph:** `/weight_graph`""")
        return
    
    try:
        weight = float(context.args[0])
        
        if weight < 30 or weight > 300:
            await update.message.reply_text("❌ Weight must be between 30-300kg!")
            return
        
        # Log weight to database
        db.log_weight(user_id, weight)
        
        # Get weight history for comparison
        weight_history = db.get_weight_history(user_id, days=30)
        
        response = f"""✅ **WEIGHT LOGGED!**

**Today's Weight:** {weight}kg

"""
        
        # Show progress if previous data exists
        if len(weight_history) > 1:
            previous_weight = weight_history[1]['weight']  # Second most recent
            change = weight - previous_weight
            
            if abs(change) >= 0.1:
                if change > 0:
                    response += f"📈 **Change:** +{change:.1f}kg since last log\n"
                else:
                    response += f"📉 **Change:** {change:.1f}kg since last log\n"
            else:
                response += "➡️ **Change:** Stable weight\n"
        
        response += """
**📊 Tracking Tips:**
• Weigh yourself same time daily
• Best time: Morning after bathroom
• Track trends, not daily fluctuations

📈 **View progress:** `/weight_graph`"""
        
        await update.message.reply_text(response)
        
    except ValueError:
        await update.message.reply_text("❌ Use numbers only! Example: `/weight 75`")

async def weight_graph_command(update: Update, context):
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    
    weight_history = db.get_weight_history(user_id, days=30)
    
    if len(weight_history) < 2:
        await update.message.reply_text("❌ Need at least 2 weight entries for graph! Use `/weight [kg]` to log weight.")
        return
    
    try:
        # Generate weight progress graph
        filename = GraphGenerator.create_weight_progress_graph(weight_history, user_name)
        
        if filename:
            with open(filename, 'rb') as graph_file:
                await update.message.reply_photo(
                    photo=graph_file,
                    caption=f"⚖️ **{user_name.upper()}'S WEIGHT PROGRESS** 📊\n\nYour weight journey visualized! 💪"
                )
            
            # Clean up file
            os.remove(filename)
        else:
            await update.message.reply_text("❌ Error generating weight graph")
            
    except Exception as e:
        await update.message.reply_text(f"❌ Error creating graph: {str(e)}")

async def calculate_1rm_command(update: Update, context):
    if len(context.args) != 3:
        await update.message.reply_text("""🧮 **1RM CALCULATOR**

Format: `/1rm [weight] [reps] [rpe]`

Example: `/1rm 100 5 8`""")
        return
    
    try:
        weight = float(context.args[0])
        reps = int(context.args[1])
        rpe = int(context.args[2])
        
        if not 1 <= rpe <= 10:
            await update.message.reply_text("❌ RPE must be 1-10!")
            return
        
        estimated_1rm = calculate_1rm(weight, reps, rpe)
        
        response = f"""🧮 **1RM ESTIMATE**

**Input**: {weight}kg x {reps} @{rpe} RPE
**Estimated 1RM**: {estimated_1rm}kg

**Training %s:**
• 90%: {estimated_1rm * 0.9:.1f}kg
• 85%: {estimated_1rm * 0.85:.1f}kg  
• 80%: {estimated_1rm * 0.8:.1f}kg
• 75%: {estimated_1rm * 0.75:.1f}kg

📝 **Log this set:** `/log [exercise] {weight} {reps} {rpe}`"""
        
        await update.message.reply_text(response)
        
    except ValueError:
        await update.message.reply_text("❌ Use numbers only!")

async def workout_generator_command(update: Update, context):
    if not context.args:
        keyboard = [
            [InlineKeyboardButton("🫸 Push Day", callback_data="workout_push")],
            [InlineKeyboardButton("🫷 Pull Day", callback_data="workout_pull")], 
            [InlineKeyboardButton("🦵 Leg Day", callback_data="workout_legs")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🔥 **CHOOSE YOUR WORKOUT:**",
            reply_markup=reply_markup
        )
        return
    
    workout_type = context.args[0].lower()
    
    workouts = {
        "push": [
            "Bench Press: 4 sets x 8-10 @8 RPE",
            "Incline DB Press: 3 sets x 10-12 @8 RPE", 
            "Shoulder Press: 3 sets x 10-12 @8 RPE",
            "Close-Grip Bench: 3 sets x 10-15 @8 RPE",
            "Lateral Raises: 3 sets x 12-15 @8 RPE",
            "Tricep Extensions: 3 sets x 12-15 @8 RPE"
        ],
        "pull": [
            "Deadlift: 4 sets x 5-6 @8 RPE",
            "Barbell Rows: 4 sets x 8-10 @8 RPE",
            "Pull-ups: 3 sets x 8-12 @8 RPE",
            "Cable Rows: 3 sets x 10-12 @8 RPE", 
            "Barbell Curls: 3 sets x 10-12 @8 RPE",
            "Hammer Curls: 3 sets x 12-15 @8 RPE"
        ],
        "legs": [
            "Squat: 4 sets x 8-10 @8 RPE",
            "Romanian Deadlift: 3 sets x 10-12 @8 RPE",
            "Bulgarian Split Squats: 3 sets x 10 each @8 RPE",
            "Leg Press: 3 sets x 12-15 @8 RPE",
            "Leg Curls: 3 sets x 12-15 @8 RPE",
            "Calf Raises: 4 sets x 15-20 @8 RPE"
        ]
    }
    
    if workout_type not in workouts:
        await update.message.reply_text("❌ Use: push, pull, or legs")
        return
    
    workout = workouts[workout_type]
    
    response = f"🔥 **{workout_type.upper()} DAY** 🔥\n\n"
    for i, exercise in enumerate(workout, 1):
        response += f"{i}. {exercise}\n"
    
    response += "\n💡 **New Features:**\n"
    response += f"🎥 Get form videos: `/video [exercise]`\n"
    response += f"📝 Log your sets: `/log [exercise] [weight] [reps] [rpe]`\n"
    response += f"📊 Track progress: `/progress [exercise]`\n"
    
    if workout_type == "push":
        response += "\n🎯 Focus on pressing power and shoulder stability!"
    elif workout_type == "pull":
        response += "\n🎯 Focus on lat engagement and posterior chain!"
    else:
        response += "\n🎯 Focus on hip hinge patterns and glute activation!"
    
    await update.message.reply_text(response)

async def button_handler(update: Update, context):
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith("workout_"):
        workout_type = query.data.replace("workout_", "")
        
        workouts = {
            "push": [
                "Bench Press: 4 sets x 8-10 @8 RPE",
                "Incline DB Press: 3 sets x 10-12 @8 RPE", 
                "Shoulder Press: 3 sets x 10-12 @8 RPE",
                "Close-Grip Bench: 3 sets x 10-15 @8 RPE",
                "Lateral Raises: 3 sets x 12-15 @8 RPE",
                "Tricep Extensions: 3 sets x 12-15 @8 RPE"
            ],
            "pull": [
                "Deadlift: 4 sets x 5-6 @8 RPE",
                "Barbell Rows: 4 sets x 8-10 @8 RPE",
                "Pull-ups: 3 sets x 8-12 @8 RPE",
                "Cable Rows: 3 sets x 10-12 @8 RPE", 
                "Barbell Curls: 3 sets x 10-12 @8 RPE",
                "Hammer Curls: 3 sets x 12-15 @8 RPE"
            ],
            "legs": [
                "Squat: 4 sets x 8-10 @8 RPE",
                "Romanian Deadlift: 3 sets x 10-12 @8 RPE",
                "Bulgarian Split Squats: 3 sets x 10 each @8 RPE",
                "Leg Press: 3 sets x 12-15 @8 RPE",
                "Leg Curls: 3 sets x 12-15 @8 RPE",
                "Calf Raises: 4 sets x 15-20 @8 RPE"
            ]
        }
        
        workout = workouts.get(workout_type, [])
        
        response = f"🔥 **{workout_type.upper()} DAY** 🔥\n\n"
        for i, exercise in enumerate(workout, 1):
            response += f"{i}. {exercise}\n"
        
        response += f"\n💪 Crush this {workout_type} session!"
        response += f"\n📝 Log sets: `/log [exercise] [weight] [reps] [rpe]`"
        response += f"\n🎥 Form help: `/video [exercise]`"
        
        await query.edit_message_text(response)

async def stats_update(update: Update, context):
    if len(context.args) < 2:
        await update.message.reply_text("""⚙️ **QUICK STATS UPDATE**

Format: `/stats [weight] [goal]`

Examples:
• `/stats 75 bulk`
• `/stats 80 cut`

**For full profile:** `/profile`""")
        return
    
    try:
        weight = float(context.args[0])
        goal = " ".join(context.args[1:]).lower()
        
        user_id = update.effective_user.id
        db.update_user_stats(user_id, weight=weight, goal=goal)
        
        response = f"""⚙️ **STATS UPDATED!**

**Weight**: {weight}kg
**Goal**: {goal.title()}

**Next steps:**
• `/workout push` - Get a workout
• `/nutrition` - See your macros
• `/log bench press 100 5 8` - Log sets

Let's get after it! 💪"""
        
        await update.message.reply_text(response)
        
    except ValueError:
        await update.message.reply_text("❌ Use: `/stats 75 bulk`")

async def nutrition_targets_command(update: Update, context):
    user_id = update.effective_user.id
    profile = db.get_user_profile(user_id)
    
    weight = profile['weight'] if profile['weight'] > 0 else 70
    goal = profile['goal']
    
    # Calculate macros based on goal
    if 'bulk' in goal:
        calories = weight * 35
        protein = weight * 2.2
        fat = weight * 1.0
    elif 'cut' in goal:
        calories = weight * 28
        protein = weight * 2.5
        fat = weight * 0.8
    else:
        calories = weight * 32
        protein = weight * 2.0
        fat = weight * 1.0
    
    carbs = (calories - (protein * 4) - (fat * 9)) / 4
    
    response = f"""🍖 **NUTRITION TARGETS** ({weight}kg, {goal.upper()})

**Daily Targets:**
• 🔥 Calories: {calories:.0f}
• 🥩 Protein: {protein:.0f}g 
• 🍞 Carbs: {carbs:.0f}g
• 🥑 Fat: {fat:.0f}g

**Simple Meal Ideas:**
• Breakfast: Oats + banana + protein powder
• Lunch: Chicken + rice + vegetables  
• Dinner: Fish + sweet potato + salad
• Snack: Greek yogurt + nuts

**💧 Hydration:** 3-4L water daily!

**Track meals:** `/food [description]`
**View progress:** `/nutrition`"""
    
    await update.message.reply_text(response)

async def motivational_chat(update: Update, context):
    user_message = update.message.text.lower()
    user_name = update.effective_user.first_name
    
    # Enhanced motivational responses
    if any(word in user_message for word in ['tired', 'exhausted', 'burnt out', 'sore']):
        responses = [
            f"Rest and recovery are when you grow stronger {user_name}! 💪 Try logging it: `/weight [kg]` and `/steps [number]`",
            f"Listen to your body! 😤 Maybe check your nutrition: `/nutrition`",
            f"Recovery is part of the process! 🌱 Track your weight and steps to monitor progress."
        ]
    elif any(word in user_message for word in ['motivated', 'pumped', 'ready', 'let\'s go']):
        responses = [
            f"THAT'S THE ENERGY! 🔥 Get a workout: `/workout push` and crush it!",
            f"YESSS {user_name}! 💪 What are we training today? Try `/workout pull`!",
            f"I LOVE IT! 🚀 Log your food first: `/food [meal]` then let's TRAIN!"
        ]
    elif any(word in user_message for word in ['pr', 'personal record', 'new max']):
        responses = [
            f"NEW PR ALERT! 🎉🔥 Log that beast: `/log [exercise] [weight] [reps] [rpe]`!",
            f"PR CITY! 🏆 Get a progress graph: `/progress [exercise]`!",
            f"BOOM! 💥 That's what I'm talking about! Log it and track that progress!"
        ]
    elif any(word in user_message for word in ['help', 'confused', 'how']):
        responses = [
            f"I'm here to help {user_name}! 🤝 Try `/help` to see all commands!",
            f"No worries! 💯 Use `/profile` to set up, then `/workout` to train!",
            f"Let's get you sorted! 📋 Start with `/help` for all features!"
        ]
    elif any(word in user_message for word in ['food', 'eat', 'nutrition', 'diet']):
        responses = [
            f"Nutrition is KEY! 🍖 Log your meals: `/food [description]`",
            f"Fuel those gains! 🔥 See your targets: `/nutrition`",
            f"Food = Fuel! 🚀 Track with `/food` and monitor with `/nutrition`"
        ]
    elif any(word in user_message for word in ['steps', 'walking', 'cardio']):
        responses = [
            f"Movement matters! 👟 Log your steps: `/steps [number]`",
            f"Every step counts! 🚶‍♂️ Target: 8k+ daily steps!",
            f"Active recovery! 💨 Track with `/steps` and see weekly progress!"
        ]
    else:
        responses = [
            f"What's good {user_name}! 💪 Ready to track some progress?",
            f"Iron Buddy Advanced here! 🔥 Try the new features: `/food`, `/steps`, `/video`!",
            f"Yo {user_name}! 🤝 Need help? `/help` shows all the new commands!",
            f"Always here for you! 💯 Your data is saved forever in the database now!"
        ]
    
    response = random.choice(responses)
    await update.message.reply_text(response)

# ==================== MAIN FUNCTION ====================

def main():
    print("🚀 Starting Iron Buddy Advanced Bot...")
    
    try:
        # Create application
        application = Application.builder().token(BOT_TOKEN).build()
        print("✅ Telegram application created")
        
        # Add all command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        
        # Profile & Stats
        application.add_handler(CommandHandler("profile", profile_command))
        application.add_handler(CommandHandler("stats", stats_update))
        
        # Workout Tracking
        application.add_handler(CommandHandler("log", log_workout_command))
        application.add_handler(CommandHandler("history", workout_history_command))
        application.add_handler(CommandHandler("progress", progress_graph_command))
        application.add_handler(CommandHandler("1rm", calculate_1rm_command))
        application.add_handler(CommandHandler("workout", workout_generator_command))
        
        # Exercise Videos
        application.add_handler(CommandHandler("video", exercise_video_command))
        application.add_handler(CommandHandler("videos", list_videos_command))
        
        # Nutrition Tracking
        application.add_handler(CommandHandler("food", food_logging_command))
        application.add_handler(CommandHandler("nutrition", daily_nutrition_command))
        application.add_handler(CommandHandler("nutrition_week", weekly_nutrition_command))
        application.add_handler(CommandHandler("targets", nutrition_targets_command))
        
        # Step Tracking
        application.add_handler(CommandHandler("steps", steps_logging_command))
        application.add_handler(CommandHandler("steps_week", weekly_steps_command))
        
        # Weight Tracking
        application.add_handler(CommandHandler("weight", weight_logging_command))
        application.add_handler(CommandHandler("weight_graph", weight_graph_command))
        
        # Button handler
        application.add_handler(CallbackQueryHandler(button_handler))
        
        # Motivational chat (must be last)
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, motivational_chat))
        
        print("✅ All handlers added successfully")
        
        # Missing numpy import for graphs - add this at the top
        try:
            import numpy as np
            print("✅ Numpy available for trend lines")
        except ImportError:
            print("⚠️  Numpy not available - install with: pip install numpy")
        
        # Start bot
        print("\n" + "="*60)
        print("🔥 IRON BUDDY ADVANCED (@Ironbuddybot) IS LIVE! 🔥")
        print("💪 NEW FEATURES: SQLite DB, Exercise Videos, Food Logging!")
        print("📊 Step Tracking, Progress Graphs, Complete Analytics!")
        print("🚀 Go to Telegram and send /start to see new features!")
        print("="*60)
        print("\n⚠️  Keep this window OPEN")
        print("⚠️  Press Ctrl+C to stop")
        print("📁 Database: iron_buddy.db")
        print("📊 Graphs saved temporarily then sent to users\n")
        
        application.run_polling()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n📋 Installation requirements:")
        print("pip install python-telegram-bot matplotlib requests numpy")
        input("Press Enter to exit...")

# This MUST be at the very end of the file
if __name__ == "__main__":
    main()