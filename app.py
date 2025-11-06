import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
from gensim.models import KeyedVectors
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from difflib import get_close_matches
import os

# PAGE CONFIG
st.set_page_config(
    page_title="HairCare Recommender - Enhanced",
    page_icon="💇‍♀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .alert-success {
        background: #10b981;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        animation: slideDown 0.3s ease-out;
    }

    .alert-warning {
        background: #f59e0b;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
        animation: slideDown 0.3s ease-out;
    }

    .alert-info {
        background: #3b82f6;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        animation: slideDown 0.3s ease-out;
    }

    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .filter-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.3rem;
    }

    .badge-strict {
        background: #10b981;
        color: white;
    }

    .badge-relaxed-1 {
        background: #f59e0b;
        color: white;
    }

    .badge-relaxed-2 {
        background: #ef4444;
        color: white;
    }

    .product-card-small {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 3px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        border: 2px solid transparent;
    }

    .product-card-small:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        border-color: #667eea;
    }

    .product-card-small.relaxed {
        border-left: 4px solid #f59e0b;
    }

    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
    }

    .stats-number {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stats-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }

    .review-card {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 8px;
    }

    .review-user {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
    }

    .review-text {
        color: #333;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }

    .review-recommend {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .recommend-yes { background: #10b981; color: white; }
    .recommend-no { background: #ef4444; color: white; }

    .pref-hint {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #555;
    }

    .pref-example {
        color: #667eea;
        font-weight: 600;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


class InputExpander:
    PROBLEM_EXPANSION = {
        'ketombe': ['anti dandruff', 'scalp treatment','ketombe basah','ketombe kering','anti ketombe','jamur'],
        'rontok': ['hair fall', 'strengthen','hair loss'],
        'kering': ['dry hair', 'frizzy','greasy', 'moisturizing', 'moisturize','moist'],
        'berminyak': ['oily hair', 'oil control','lepek','kusam'],
        'bercabang': ['split ends','ujung rambut'],
    }

    @staticmethod
    def expand_problems(problems: List[str]) -> List[str]:
        expanded = problems.copy()
        for problem in problems:
            if problem in InputExpander.PROBLEM_EXPANSION:
                expanded.extend(InputExpander.PROBLEM_EXPANSION[problem])
        return expanded

class PreferenceParser:
    @staticmethod
    def parse_price(text: str) -> Optional[Tuple[float, float]]:
        text = text.lower()

        pattern_under = r'(?:dibawah|under|kurang dari|<)\s*(\d+)(?:rb|ribu|k)?'
        match = re.search(pattern_under, text)
        if match:
            val = int(match.group(1))
            if val < 1000:
                val *= 1000
            return (0, val)

        pattern_over = r'(?:diatas|over|lebih dari|>)\s*(\d+)(?:rb|ribu|k)?'
        match = re.search(pattern_over, text)
        if match:
            val = int(match.group(1))
            if val < 1000:
                val *= 1000
            return (val, float('inf'))

        pattern_range = r'(\d+)(?:rb|ribu|k)?\s*(?:-|sampai|hingga)\s*(\d+)(?:rb|ribu|k)?'
        match = re.search(pattern_range, text)
        if match:
            min_val = int(match.group(1))
            max_val = int(match.group(2))
            if min_val < 1000:
                min_val *= 1000
            if max_val < 1000:
                max_val *= 1000
            return (min_val, max_val)

        if any(kw in text for kw in ['murah', 'budget', 'affordable', 'terjangkau']):
            return (0, 100000)
        if any(kw in text for kw in ['sedang', 'moderate', 'menengah']):
            return (100000, 300000)
        if any(kw in text for kw in ['mahal', 'premium', 'mehong', 'mewah']):
            return (300000, float('inf'))

        return None

    @staticmethod
    def parse_rating(text: str) -> Optional[float]:
        text = text.lower()

        pattern = r'(?:rating|bintang)?\s*(\d+(?:\.\d+)?)\s*(?:keatas|ke atas|up|\+)?'
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))

        if any(kw in text for kw in ['rating tinggi', 'rating bagus', 'highly rated']):
            return 4.5
        if any(kw in text for kw in ['rating ok', 'rating cukup']):
            return 4.0

        return None

    @staticmethod
    def parse_ingredients(text: str) -> List[str]:
        ingredients = []
        text = text.lower()

        ingredient_patterns = [
            'argan oil', 'coconut oil', 'jojoba oil', 'olive oil',
            'keratin', 'collagen', 'biotin', 'caffeine',
            'vitamin e', 'vitamin b5', 'panthenol',
            'aloe vera', 'tea tree', 'ginseng',
            'sulfate free', 'paraben free', 'silicone free',
            'natural', 'organic', 'herbal'
        ]

        for ingredient in ingredient_patterns:
            if ingredient in text:
                ingredients.append(ingredient)

        return ingredients

    @staticmethod
    def parse_size(text: str) -> Optional[Tuple[float, float]]:
        text = text.lower()

        pattern_exact = r'(\d+)\s*ml'
        match = re.search(pattern_exact, text)
        if match:
            val = float(match.group(1))
            return (val * 0.8, val * 1.2)

        pattern_over = r'(?:diatas|over|>)\s*(\d+)\s*ml'
        match = re.search(pattern_over, text)
        if match:
            val = float(match.group(1))
            return (val, float('inf'))

        if any(kw in text for kw in ['ukuran besar', 'large size', 'jumbo']):
            return (300, float('inf'))
        if any(kw in text for kw in ['ukuran kecil', 'travel size', 'mini']):
            return (0, 100)

        return None

    @staticmethod
    def parse_brand(text: str, available_brands: set) -> Tuple[Optional[str], bool]:
        if not text:
            return (None, True)

        text = text.lower().strip()

        negative_keywords = [
            'jangan', 'gak mau', 'gamau', 'tidak mau', 'bukan', 'gak',
            'selain', 'kecuali', 'hindari', 'avoid', 'except','enggak',
            "don't want", 'no', 'tanpa', 'exclude', 'yang lain'
        ]

        is_negative = any(neg in text for neg in negative_keywords)
        detected_brand = None

        if text in available_brands:
            detected_brand = text
        else:
            for brand in available_brands:
                pattern = r'\b' + re.escape(brand) + r'\b'
                if re.search(pattern, text):
                    detected_brand = brand
                    break

            if not detected_brand:
                for brand in available_brands:
                    if brand in text or text in brand:
                        if len(brand) > 3 or len(text) > 3:
                            detected_brand = brand
                            break

            if not detected_brand and not is_negative:
                words = text.split()
                for word in words:
                    if len(word) > 3:
                        matches = get_close_matches(word, available_brands, n=1, cutoff=0.75)
                        if matches:
                            detected_brand = matches[0]
                            break

            if not detected_brand:
                for brand in available_brands:
                    if text in brand and len(text) > 4:
                        detected_brand = brand
                        break

        if detected_brand:
            return (detected_brand, not is_negative)

        return (None, True)

class UserInputVectorizer:
    def __init__(self, tfidf_vectorizer_path, fasttext_finetuned_path, products_df=None):
        with open(tfidf_vectorizer_path, 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)

        self.fasttext_model = KeyedVectors.load_word2vec_format(
            fasttext_finetuned_path, binary=False, encoding='utf-8',
            unicode_errors='ignore', limit=None
        )

        if products_df is not None:
            self.brand_vocab = set(products_df['brand'].str.lower().unique())
            self.product_vocab = set()
            for col in ['product_name', 'claim']:
                if col in products_df.columns:
                    for text in products_df[col].fillna(''):
                        self.product_vocab.update(str(text).lower().split())
        else:
            self.brand_vocab = set()
            self.product_vocab = set()

    def correct_typo(self, word: str) -> str:
        if len(word) < 3 or word in self.product_vocab or word in self.brand_vocab:
            return word

        matches = get_close_matches(word, self.product_vocab, n=1, cutoff=0.80)
        if matches:
            return matches[0]
        return word

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.split()

        if self.product_vocab:
            corrected = [self.correct_typo(w) for w in words if len(w) > 1]
            return ' '.join(corrected)
        return re.sub(r'\s+', ' ', text).strip()

    def vectorize(self, user_input):
        if isinstance(user_input, list):
            user_input = ' '.join(user_input)

        cleaned = self.preprocess(user_input)

        tfidf_vec = self.tfidf_vectorizer.transform([cleaned]).toarray()[0]
        tfidf_norm = normalize([tfidf_vec], norm='l2')[0]

        words = cleaned.split()
        word_vecs = [self.fasttext_model[w] for w in words if w in self.fasttext_model]
        fasttext_vec = np.mean(word_vecs, axis=0) if word_vecs else np.zeros(self.fasttext_model.vector_size)
        fasttext_norm = normalize([fasttext_vec], norm='l2')[0]

        combined = np.concatenate([tfidf_norm * 0.4, fasttext_norm * 0.6])
        return normalize([combined], norm='l2')[0]

class HaircareRecommender:
    def __init__(self,
                 products_path='dataset/preprocess/product_info.csv',
                 product_vectors_path='models/product_combined_vectors_finetuned.npy',
                 tfidf_vectorizer_path='models/tfidf_vectorizer.pkl',
                 fasttext_finetuned_path='models/fasttext_haircare_gensim.vec'):

        self.products = pd.read_csv(products_path)
        self.product_vectors = np.load(product_vectors_path)
        self.vectorizer = UserInputVectorizer(
            tfidf_vectorizer_path, fasttext_finetuned_path, self.products
        )

        self.category_mapping = {
            'mask': 'mask', 'hair mask': 'mask', 'hair_mask': 'mask',
            'serum': 'hair serum', 'hair serum': 'hair serum', 'hair_serum': 'hair serum',
            'oil': 'hair oil', 'hair oil': 'hair oil', 'hair_oil': 'hair oil',
            'shampoo': 'shampoo', 'conditioner': 'conditioner',
        }

        self.filter_logs = []
        self.brand_boost_applied = False

    def normalize_category(self, cat: str) -> str:
        return self.category_mapping.get(cat.lower().strip(), cat.lower().strip())

    def apply_filters(self, df: pd.DataFrame, filters: Dict, relax_level: int = 0) -> Tuple[pd.DataFrame, Dict]:
        filtered = df.copy()
        strict_count = len(df)
        filter_messages = []

        if filters.get('brand') and filters.get('brand_preference') == 'negative':
            brand = filters['brand']
            before_count = len(filtered)
            filtered = filtered[filtered['brand'].str.lower() != brand]
            after_count = len(filtered)

            if before_count > after_count:
                filter_messages.append(
                    f"Excluded brand: {brand.upper()} ({before_count - after_count} products removed)")

        if filters.get('brand') and filters.get('brand_preference') == 'positive':
            brand = filters['brand']
            filter_messages.append(f"Brand preference: {brand.upper()} (will boost similarity)")

        if filters.get('price_range'):
            min_price, max_price = filters['price_range']

            if relax_level == 1:
                if max_price != float('inf'):
                    max_price *= 1.2
                if min_price > 0:
                    min_price *= 0.8
                filter_messages.append(f"Price filter relaxed 20%: Rp {min_price:,.0f} - Rp {max_price:,.0f}")
            elif relax_level == 2:
                if max_price != float('inf'):
                    max_price *= 1.4
                if min_price > 0:
                    min_price *= 0.6
                filter_messages.append(f"Price filter relaxed 40%: Rp {min_price:,.0f} - Rp {max_price:,.0f}")
            else:
                filter_messages.append(f"Price filter (strict): Rp {min_price:,.0f} - Rp {max_price:,.0f}")

            filtered = filtered[
                (filtered['price'] >= min_price) &
                (filtered['price'] <= max_price)
                ]

        if filters.get('min_rating'):
            min_rating = filters['min_rating']

            if relax_level == 1:
                min_rating -= 0.3
                filter_messages.append(f"Rating filter relaxed: ≥ {min_rating:.1f}")
            elif relax_level == 2:
                min_rating -= 0.5
                filter_messages.append(f"Rating filter relaxed: ≥ {min_rating:.1f}")
            else:
                filter_messages.append(f"Rating filter (strict): ≥ {min_rating}")

            filtered = filtered[filtered['rating'] >= min_rating]

        if filters.get('size_range'):
            min_size, max_size = filters['size_range']

            if relax_level >= 1:
                if max_size != float('inf'):
                    max_size *= 1.5
                if min_size > 0:
                    min_size *= 0.5
                filter_messages.append(f"Size filter relaxed: {min_size:.0f} - {max_size:.0f} ml")
            else:
                filter_messages.append(f"Size filter (strict): {min_size:.0f} - {max_size:.0f} ml")

            filtered = filtered[
                (filtered['size_value'] >= min_size) &
                (filtered['size_value'] <= max_size)
                ]

        if filters.get('ingredients'):
            ingredients = filters['ingredients']
            for ingredient in ingredients:
                if 'ingredients' in filtered.columns:
                    mask = filtered['ingredients'].fillna('').str.lower().str.contains(ingredient, na=False)
                    filtered = filtered[mask]
            filter_messages.append(f"Ingredient filter: {', '.join(ingredients)}")

        relaxed_count = len(filtered)
        coverage_metrics = {
            'strict_count': strict_count if relax_level == 0 else 0,
            'relaxed_count': relaxed_count,
            'relax_level': relax_level,
            'filter_messages': filter_messages
        }

        return filtered, coverage_metrics

    def recommend(self, user_name, problems, categories, preferences, top_n):
        if categories and len(categories) > 3:
            categories = categories[:3]
        if top_n > 10:
            top_n = 10

        self.last_preferences = preferences or ''
        self.filter_logs = []
        self.brand_boost_applied = False

        filters = {}
        if preferences:
            self.filter_logs.append("Parsing preferences...")

            available_brands = set(self.products['brand'].str.lower().unique())
            brand_result = PreferenceParser.parse_brand(preferences, available_brands)

            if brand_result[0]:
                brand_name, is_positive = brand_result

                if is_positive:
                    filters['brand'] = brand_name
                    filters['brand_preference'] = 'positive'
                    self.filter_logs.append(f"Brand preference detected: {brand_name.upper()} (BOOST)")
                else:
                    filters['brand'] = brand_name
                    filters['brand_preference'] = 'negative'
                    self.filter_logs.append(f"Brand exclusion detected: {brand_name.upper()} (AVOID)")

            price_range = PreferenceParser.parse_price(preferences)
            if price_range:
                filters['price_range'] = price_range

            min_rating = PreferenceParser.parse_rating(preferences)
            if min_rating:
                filters['min_rating'] = min_rating

            ingredients = PreferenceParser.parse_ingredients(preferences)
            if ingredients:
                filters['ingredients'] = ingredients

            size_range = PreferenceParser.parse_size(preferences)
            if size_range:
                filters['size_range'] = size_range

            if filters:
                self.filter_logs.append(f"Detected {len(filters)} filter(s)")

        working_df = self.products.copy()

        # CRITICAL: If user specified categories, filter by category FIRST
        if categories:
            working_df = working_df[working_df['category'].isin(categories)]
            self.filter_logs.append(f"Filtered by category: {', '.join(categories)} → {len(working_df)} products")

        coverage_metrics = None

        if filters:
            self.filter_logs.append("Applying filters with adaptive relaxation...")

            working_df, coverage_metrics = self.apply_filters(working_df.copy(), filters, relax_level=0)
            self.filter_logs.append(f"📊 Strict filter results: {len(working_df)} products")

            if len(working_df) < top_n:
                self.filter_logs.append(f"Insufficient results ({len(working_df)} < {top_n})")
                self.filter_logs.append(f"Relaxing filters by 20%...")

                base_df = self.products.copy()
                if categories:
                    base_df = base_df[base_df['category'].isin(categories)]

                working_df, coverage_metrics = self.apply_filters(base_df, filters, relax_level=1)
                self.filter_logs.append(f"Relaxed filter results: {len(working_df)} products")

            if len(working_df) < top_n:
                self.filter_logs.append(f"Still insufficient ({len(working_df)} < {top_n})")
                self.filter_logs.append(f"Relaxing filters by 40%...")

                base_df = self.products.copy()
                if categories:
                    base_df = base_df[base_df['category'].isin(categories)]

                working_df, coverage_metrics = self.apply_filters(base_df, filters, relax_level=2)
                self.filter_logs.append(f"Final results: {len(working_df)} products")

            if working_df.empty:
                self.filter_logs.append("No products match even with relaxed filters")
                return pd.DataFrame()

        expanded_problems = InputExpander.expand_problems(problems)
        user_input = expanded_problems.copy()
        if categories:
            user_input.extend(categories)
        if preferences:
            user_input.append(preferences)

        user_vector = self.vectorizer.vectorize(user_input)

        filtered_indices = working_df.index.tolist()
        filtered_vectors = self.product_vectors[filtered_indices]
        similarities = cosine_similarity([user_vector], filtered_vectors)[0]

        if categories:
            for i, idx in enumerate(filtered_indices):
                if working_df.loc[idx, 'category'] in categories:
                    similarities[i] = min(similarities[i] + 0.05, 1.0)

        if filters.get('brand'):
            brand = filters['brand']
            brand_preference = filters.get('brand_preference', 'positive')

            if brand_preference == 'positive':
                for i, idx in enumerate(filtered_indices):
                    if working_df.loc[idx, 'brand'].lower() == brand:
                        similarities[i] = min(similarities[i] + 0.10, 1.0)

                self.brand_boost_applied = True
                self.filter_logs.append(f"Brand boost applied: {brand.upper()} (+10%)")

            elif brand_preference == 'negative':
                for i, idx in enumerate(filtered_indices):
                    if working_df.loc[idx, 'brand'].lower() == brand:
                        similarities[i] = max(similarities[i] * 0.50, 0)

                self.brand_boost_applied = True
                self.filter_logs.append(f"Brand penalty applied: {brand.upper()} (-50%)")

        # Simple top-N selection - no category splitting
        top_local_indices = similarities.argsort()[-top_n:][::-1]
        results = [
            {
                'idx': int(filtered_indices[i]),
                'similarity': float(similarities[i]),
                'is_strict_match': coverage_metrics['relax_level'] == 0 if coverage_metrics else True,
                'relax_level': coverage_metrics['relax_level'] if coverage_metrics else 0
            }
            for i in top_local_indices
        ]

        return self._format_results_from_indices(results, problems, coverage_metrics)

    def _parse_reviews(self, reviews_raw: str) -> List[Dict]:
        if pd.isna(reviews_raw) or not str(reviews_raw).strip():
            return []

        reviews = []
        try:
            review_items = str(reviews_raw).split('||')

            for review_item in review_items:
                if not review_item.strip():
                    continue

                parts = review_item.split('|')
                review_dict = {}

                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        review_dict[key.strip()] = value.strip()

                if review_dict:
                    reviews.append({
                        'user': review_dict.get('user', 'Anonymous'),
                        'user_url': review_dict.get('user_URL', ''),
                        'review_text': review_dict.get('text', 'No review text'),
                        'recommend': review_dict.get('recommend', 'unknown').lower()
                    })
        except Exception as e:
            return []

        return reviews

    def _format_results_from_indices(self, results, problems, coverage_metrics):
        formatted = []
        for item in results:
            idx = item['idx']
            product = self.products.iloc[idx]

            desc = product.get('claim', '')
            if pd.isna(desc) or not desc:
                desc = str(product.get('text_embedding', ''))[:150] + '...'

            reviews_parsed = self._parse_reviews(product.get('reviews', ''))

            formatted.append({
                'product_id': idx,
                'brand': product['brand'],
                'name': product['product_name'],
                'category': product['category'],
                'rating': product['rating'],
                'total_reviews': product.get('total_reviews', 0),
                'price': product['price'],
                'size': f"{product.get('size_value', 0):.0f} {product.get('size_unit', 'ml')}",
                'similarity': item['similarity'],
                'description': desc,
                'product_url': product.get('product_URL', ''),
                'ingredients': product.get('ingredients', ''),
                'how_to_use': product.get('use', ''),
                'image_url': product.get('img_URL', ''),
                'bpom': product.get('bpom', 'N/A'),
                'reviews': reviews_parsed,
                'user_problems': ', '.join(problems),
                'user_preferences': self.last_preferences,
                'is_strict_match': item.get('is_strict_match', True),
                'relax_level': item.get('relax_level', 0)
            })

        df_result = pd.DataFrame(formatted)

        if coverage_metrics:
            df_result.attrs['coverage_metrics'] = coverage_metrics

        return df_result

@st.cache_resource
def load_recommender():
    try:
        recommender = HaircareRecommender(
            products_path='dataset/preprocess/product_info.csv',
            product_vectors_path='models/product_combined_vectors_finetuned.npy',
            tfidf_vectorizer_path='models/tfidf_vectorizer.pkl',
            fasttext_finetuned_path='models/fasttext_haircare_gensim.vec'
        )
        return recommender
    except Exception as e:
        st.error(f"❌ Error loading recommender: {e}")
        return None

recommender = load_recommender()

selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Rekomendasi", "About"],
    icons=["house-fill", "search-heart", "info-circle-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

def show_alert(message, alert_type="success"):
    alert_class = f"alert-{alert_type}"
    st.markdown(f'<div class="{alert_class}">🎉 {message}</div>', unsafe_allow_html=True)

def save_recommendations_to_csv(df, user_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recommendations_{user_name.replace(' ', '_')}_{timestamp}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    return filename

if selected == "Dashboard":
    st.markdown('<h1 style="text-align:center;color:white;">💇‍♀️ HairCare Recommender System</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:white;font-size:1.2rem;margin-bottom:3rem;">Sistem Rekomendasi Produk Perawatan Rambut dengan Adaptive Filtering</p>',
        unsafe_allow_html=True)

    st.markdown('<h2 style="color:white;text-align:center;margin-bottom:2rem;">Kenali Masalah Rambutmu</h2>',
                unsafe_allow_html=True)

    problems = [
        {
            "title": "Rambut Kering & Kusam",
            "desc": "Rambut kering terjadi ketika rambut kehilangan kelembaban alami. Penyebabnya bisa dari paparan sinar matahari, penggunaan alat styling panas berlebihan, atau kurangnya perawatan yang tepat. Solusinya adalah menggunakan hair oil atau hair mask yang kaya nutrisi untuk mengembalikan kelembaban rambut."
        },
        {
            "title": "Rambut Berminyak & Lepek",
            "desc": "Produksi sebum berlebih di kulit kepala menyebabkan rambut cepat berminyak dan lepek. Kondisi ini bisa dipicu oleh hormon, cuaca panas, atau penggunaan produk yang tidak cocok. Gunakan shampoo khusus untuk rambut berminyak dengan formula yang ringan namun efektif membersihkan."
        },
        {
            "title": "Rambut Rontok",
            "desc": "Kerontokan rambut normal terjadi 50-100 helai per hari. Namun jika berlebihan, bisa disebabkan oleh stress, kekurangan nutrisi, atau masalah hormon. Treatment intensif dengan serum anti-rontok dan vitamin rambut dapat membantu memperkuat akar rambut dan mengurangi kerontokan."
        },
        {
            "title": "Ketombe & Kulit Kepala Gatal",
            "desc": "Ketombe disebabkan oleh pertumbuhan jamur Malassezia atau kulit kepala yang terlalu kering/berminyak. Gejalanya berupa serpihan putih dan rasa gatal. Shampoo anti-ketombe dengan kandungan zinc pyrithione atau ketoconazole efektif mengatasi masalah ini."
        },
        {
            "title": "Ujung Rambut Bercabang",
            "desc": "Split ends atau ujung rambut bercabang terjadi ketika lapisan pelindung rambut (cuticle) rusak. Penyebabnya termasuk penggunaan alat panas, pewarnaan berlebihan, atau jarang memotong rambut. Gunakan hair serum dan conditioner untuk melindungi dan nutrisi ujung rambut."
        }
    ]

    cols = st.columns(5, gap="medium")
    for i, problem in enumerate(problems):
        with cols[i]:
            st.markdown(f"""
                <div class="stats-card" style="text-align:left;height:100%;min-height:280px;">
                    <h3 style="color:#667eea;margin-bottom:1rem;font-size:1.1rem;">{problem['title']}</h3>
                    <p style="color:#666;line-height:1.6;font-size:0.9rem;">{problem['desc']}</p>
                </div>
                """, unsafe_allow_html=True)

elif selected == "Rekomendasi":
    st.markdown('<h1 style="color:white;text-align:center;">🔍 Dapatkan Rekomendasi Produk</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:white;text-align:center;font-size:1.1rem;margin-bottom:2rem;">Lengkapi form di bawah untuk mendapatkan rekomendasi produk haircare yang sesuai</p>',
        unsafe_allow_html=True)

    if recommender is None:
        st.error("❌ Backend recommender gagal dimuat. Periksa path file!")
        st.stop()

    with st.container():
        st.markdown(
            '<div style="background:white;padding:2rem;border-radius:15px;box-shadow:0 5px 20px rgba(0,0,0,0.2);">',
            unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("### Informasi Dasar")
            user_name = st.text_input(
                "Nama Anda",
                placeholder="Masukkan nama Anda",
                help="Nama akan digunakan untuk menyimpan hasil rekomendasi"
            )

            st.markdown("### Masalah Rambut")
            st.markdown(
                '<p style="font-size:0.9rem;color:#666;margin-bottom:0.5rem;">Pilih semua masalah yang sedang dialami (minimal 1)</p>',
                unsafe_allow_html=True)

            hair_problems = st.multiselect(
                "Pilih masalah rambut",
                ["ketombe", "rontok", "kering", "berminyak", "bercabang"],
                help="Pilih satu atau lebih masalah rambut",
                label_visibility="collapsed"
            )

            st.markdown("### Kategori Produk")
            st.markdown(
                '<p style="font-size:0.9rem;color:#666;margin-bottom:0.5rem;">Pilih jenis produk yang diinginkan (max 3)</p>',
                unsafe_allow_html=True)

            categories_raw = st.multiselect(
                "Pilih kategori",
                ["shampoo", "conditioner", "hair mask", "hair serum", "hair oil"],
                help="Kosongkan untuk rekomendasi dari semua kategori",
                label_visibility="collapsed"
            )

            categories = [recommender.normalize_category(cat) for cat in categories_raw] if categories_raw else None

        with col2:
            st.markdown("### Preferensi (Opsional)")
            st.markdown(
                '<p style="font-size:0.9rem;color:#666;margin-bottom:0.5rem;">Masukkan preferensi untuk filtering lebih detail</p>',
                unsafe_allow_html=True)

            preferences = st.text_area(
                "Preferensi Anda",
                placeholder="Anda bisa memasukkan preferensi tambahan mengenai produk yang Anda inginkan\nseperti harga,rating,ukuran,dll",
                height=100,
                help="Kombinasi bebas: harga, rating, ingredients, ukuran, brand (optional)",
                label_visibility="collapsed"
            )

            st.markdown("### Jumlah Rekomendasi")
            top_n = st.slider("Berapa produk yang ingin direkomendasi?", 5, 10, 10)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_btn = st.button("Dapatkan Rekomendasi", use_container_width=True, type="primary")

    if submit_btn:
        if not user_name or not user_name.strip():
            show_alert("⚠️ Mohon masukkan nama Anda", "warning")
        elif not hair_problems:
            show_alert("⚠️ Mohon pilih minimal 1 masalah rambut", "warning")
        else:
            with st.spinner("Memproses rekomendasi..."):
                results = recommender.recommend(
                    user_name=user_name,
                    problems=hair_problems,
                    categories=categories,
                    preferences=preferences.lower() if preferences else None,
                    top_n=top_n
                )

                st.session_state.recommendations = results
                st.session_state.user_name = user_name
                st.session_state.hair_problems = hair_problems
                st.session_state.preferences = preferences
                st.session_state.categories_raw = categories_raw
                st.session_state.top_n = top_n
                st.session_state.filter_logs = recommender.filter_logs
                st.session_state.brand_boost_applied = recommender.brand_boost_applied
                st.session_state.show_detail = False

    if hasattr(st.session_state, 'recommendations') and not st.session_state.recommendations.empty:
        df_recs = st.session_state.recommendations

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("---")

        success_msg = f"Berhasil menemukan {len(df_recs)} rekomendasi produk untuk {st.session_state.user_name}!"
        if st.session_state.brand_boost_applied:
            success_msg += " (Brand preference applied)"
        show_alert(success_msg, "success")

        if hasattr(st.session_state, 'filter_logs') and st.session_state.filter_logs:
            with st.expander("📋 Lihat Log Filter Process", expanded=False):
                for log in st.session_state.filter_logs:
                    st.markdown(f"- {log}")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(df_recs)}</div>
                <div class="stats-label">Produk Ditemukan</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            avg_price = df_recs['price'].mean()
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">Rp {avg_price:,.0f}</div>
                <div class="stats-label">Rata-rata Harga</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            avg_rating = df_recs['rating'].mean()
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{avg_rating:.1f}/5</div>
                <div class="stats-label">Rata-rata Rating</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            avg_sim = df_recs['similarity'].mean()
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{avg_sim:.0%}</div>
                <div class="stats-label">Avg Similarity</div>
            </div>
            """, unsafe_allow_html=True)

        if hasattr(df_recs, 'attrs') and 'coverage_metrics' in df_recs.attrs:
            metrics = df_recs.attrs['coverage_metrics']
            if metrics['relax_level'] > 0:
                strict_count = sum(df_recs['is_strict_match'])
                total_count = len(df_recs)
                coverage = (strict_count / total_count * 100) if total_count > 0 else 0

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Simpan Hasil ke CSV", use_container_width=True, type="secondary"):
                try:
                    filename = save_recommendations_to_csv(df_recs, st.session_state.user_name)
                    show_alert(f"✅ Rekomendasi berhasil disimpan ke: {filename}", "success")

                    with open(filename, 'rb') as f:
                        st.download_button(
                            label="Download File CSV",
                            data=f,
                            file_name=filename,
                            mime='text/csv',
                            use_container_width=True
                        )
                except Exception as e:
                    show_alert(f"❌ Gagal menyimpan file: {str(e)}", "warning")

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### Hasil Rekomendasi")
        st.markdown(
            f"<p style='color:white;'>Masalah: <strong>{', '.join(st.session_state.hair_problems)}</strong></p>",
            unsafe_allow_html=True)

        cols = st.columns(5, gap="medium")

        for i, (idx, row) in enumerate(df_recs.iterrows()):
            col = cols[i % 5]

            with col:
                with st.container():
                    img_url = row.get('image_url', '')
                    if img_url and str(img_url).strip() and str(img_url) != 'nan':
                        try:
                            st.image(img_url, use_column_width=True)
                        except:
                            st.markdown(
                                '<div style="height:180px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;font-size:3rem;border-radius:8px;">📦</div>',
                                unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<div style="height:180px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;font-size:3rem;border-radius:8px;">📦</div>',
                            unsafe_allow_html=True)

                    st.markdown(f"**{row['name'][:40]}...**" if len(row['name']) > 40 else f"**{row['name']}**")
                    st.markdown(f"<p style='font-size:0.85rem;color:#999;'>{row['brand']}</p>", unsafe_allow_html=True)

                    try:
                        price_str = f"Rp {int(float(row['price'])):,}"
                    except:
                        price_str = str(row['price'])

                    st.markdown(f"<p style='font-size:0.9rem;font-weight:700;color:#f5576c;'>{price_str}</p>",
                                unsafe_allow_html=True)
                    st.markdown(f"⭐ {row['rating']:.1f}/5 ({int(row['total_reviews'])} review)")

                    if row.get('relax_level', 0) > 0:
                        st.markdown(
                            f'<span class="filter-badge badge-relaxed-{row["relax_level"]}">{row["similarity"]:.0%} match</span>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="filter-badge badge-strict">{row["similarity"]:.0%} match</span>',
                                    unsafe_allow_html=True)

                    if st.button(f"Detail", key=f"detail_{i}", use_container_width=True):
                        st.session_state.selected_product_idx = i
                        st.session_state.show_detail = True
                        st.rerun()

        if hasattr(st.session_state, 'show_detail') and st.session_state.show_detail:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("---")

            if st.button("← Kembali ke Hasil", key="back_btn"):
                st.session_state.show_detail = False
                st.rerun()

            row = df_recs.iloc[st.session_state.selected_product_idx]

            col1, col2 = st.columns([1.2, 1.8], gap="large")

            with col1:
                img_url = row.get('image_url', '')
                if img_url and str(img_url).strip() and str(img_url) != 'nan':
                    st.image(img_url, use_column_width=True)
                else:
                    st.markdown(
                        '<div style="height:400px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;border-radius:15px;font-size:4rem">📦</div>',
                        unsafe_allow_html=True)

            with col2:
                st.markdown(f"## {row['name']}")
                st.markdown(f"**{row['brand']}** • {row['category']}")

                try:
                    price_str = f"Rp {int(float(row['price'])):,}"
                except:
                    price_str = str(row['price'])

                st.markdown(f"### {price_str}")
                st.markdown(f"📏 **Ukuran:** {row['size']}")
                st.markdown(f"⭐ **Rating:** {row['rating']:.1f}/5 ({int(row['total_reviews'])} reviews)")
                st.markdown(f"✨ **Kesesuaian:** {row['similarity']:.0%}")
                st.markdown(f"🏥 **BPOM:** {row.get('bpom', 'N/A')}")

                if row.get('product_url') and str(row.get('product_url')).strip() and str(
                        row.get('product_url')) != 'nan':
                    st.markdown(f"[🛒 **Beli di Merchant**]({row.get('product_url')})")

            st.markdown("---")

            tab1, tab2, tab3, tab4 = st.tabs(["📝 Deskripsi", "📖 Cara Pakai", "🧪 Ingredients", "💬 Reviews"])

            with tab1:
                st.write(row.get('description', '-') if row.get('description') else '-')

            with tab2:
                st.write(row.get('how_to_use', '-') if row.get('how_to_use') else '-')

            with tab3:
                st.write(row.get('ingredients', '-') if row.get('ingredients') else '-')

            with tab4:
                reviews = row.get('reviews', [])
                if reviews and len(reviews) > 0:
                    st.markdown(f"**{len(reviews)} Review(s)**")
                    for review in reviews:
                        recommend_class = "recommend-yes" if review['recommend'] == 'yes' else "recommend-no"
                        recommend_text = "✅ Recommended" if review['recommend'] == 'yes' else "❌ Not Recommended"

                        st.markdown(f"""
                        <div class="review-card">
                            <div class="review-user">👤 {review['user']}</div>
                            <div class="review-text">{review['review_text']}</div>
                            <span class="review-recommend {recommend_class}">{recommend_text}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Belum ada review untuk produk ini")

elif selected == "About":
    st.markdown('<h1 style="color:white;text-align:center;">Tentang Sistem</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:white;text-align:center;font-size:1.1rem;margin-bottom:2rem;">Haircare Recommender System</p>',
        unsafe_allow_html=True)

    st.markdown("""
        <div class="stats-card" style="text-align:left;">
            <h2 style="color:#667eea;">Sekilas tentang HairCare Recommender</h2>
            <p style="line-height:1.8;color:#444;font-size:1rem;">
                <strong>HairCare Recommender</strong> adalah sistem rekomendasi produk perawatan rambut berbasis 
                <em>Content-Based Filtering (CBF)</em> yang memanfaatkan pemrosesan bahasa alami 
                (<strong>FastText</strong> + <strong>TF-IDF</strong>) untuk mencocokkan kondisi rambut pengguna 
                dengan deskripsi produk. Sistem ini dirancang untuk membantu Anda memilih produk yang relevan 
                berdasarkan masalah rambut seperti <em>ketombe, rontok, kering, berminyak,</em> dan <em>bercabang</em>,
                serta mempertimbangkan preferensi pribadi seperti harga, merek, dan rating.
            </p>
            <h5 style="color:#667eea;margin-top:1rem;">Fitur Utama</h5>
            <ul style="line-height:2;color:#444;font-size:1rem;">
                <li>Rekomendasi berbasis isi produk (deskripsi & ulasan)</li>
                <li>Deteksi preferensi merek otomatis (positif / negatif)</li>
                <li>Filter adaptif dengan auto-relaxation</li>
                <li>Ekspor hasil rekomendasi beserta metadata</li>
            </ul>
            <h5 style="color:#667eea;margin-top:1rem;">Batasan Sistem</h5>
            <ul style="line-height:2;color:#444;font-size:1rem;">
                <li>Rekomendasi bersifat berbasis teks, sistem tidak memverifikasi klaim pemasaran (overclaim).</li>
                <li>Tidak semua produk memiliki daftar ingredients, sehingga validasi bahan aktif tidak selalu tersedia.</li>
                <li>Untuk rekomendasi dermatologis, disarankan berkonsultasi dengan ahli.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-card" style="text-align:left;">
        <h2 style="color:#667eea;">Cara Penggunaan</h2>
        <ol style="line-height:2;color:#666;font-size:1rem;">
            <li><strong>Input Nama:</strong> Masukkan nama Anda untuk personalisasi</li>
            <li><strong>Pilih Masalah:</strong> Pilih 1 atau lebih masalah rambut yang dialami</li>
            <li><strong>Pilih Kategori:</strong> (Opsional) Pilih jenis produk yang diinginkan (max 3 kategori). Sistem akan merekomendasikan dari kategori yang dipilih.</li>
            <li><strong>Set Preferensi:</strong> (Opsional) Brand positif/negatif, harga, rating, ingredients, ukuran</li>
            <li><strong>Dapatkan Rekomendasi:</strong> Sistem akan memberikan top-N produk terbaik sesuai kategori yang dipilih</li>
            <li><strong>Lihat Detail:</strong> Cek deskripsi produk, BPOM, ingredients, cara pakai, dan user reviews</li>
            <li><strong>Simpan Hasil:</strong> Export hasil ke CSV untuk dokumentasi</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;color:white;padding:2rem;">
        <p style="font-size:1.2rem;margin-bottom:0.5rem;">💇‍♀️ HairCare Recommender System</p>
        <p style="font-size:0.9rem;margin-top:1rem;opacity:0.8;">© 2025 | Made with ❤️ for better hair care</p>
    </div>
    """, unsafe_allow_html=True)