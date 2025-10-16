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
    page_title="HairCare Recommender - Adaptive Filters",
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

    /* Alert Styles */
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

    /* Filter Metrics Badge */
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

    /* Product Cards */
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

    /* Stats Card */
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

    /* Preference Input Hints */
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
        'ketombe': ['anti dandruff', 'scalp treatment'],
        'rontok': ['hair fall', 'strengthen'],
        'kering': ['dry hair', 'moisturizing'],
        'berminyak': ['oily hair', 'oil control'],
        'bercabang': ['split ends'],
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
        if any(kw in text for kw in ['mahal', 'premium', 'luxury', 'mewah']):
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
            'mask': 'hair mask', 'hair mask': 'hair mask', 'hair_mask': 'hair mask',
            'serum': 'hair serum', 'hair serum': 'hair serum', 'hair_serum': 'hair serum',
            'oil': 'hair oil', 'hair oil': 'hair oil', 'hair_oil': 'hair oil',
            'shampoo': 'shampoo', 'conditioner': 'conditioner',
        }

        self.filter_logs = []

    def normalize_category(self, cat: str) -> str:
        return self.category_mapping.get(cat.lower().strip(), cat.lower().strip())

    def apply_filters(self, df: pd.DataFrame, filters: Dict, relax_level: int = 0) -> Tuple[pd.DataFrame, Dict]:
        filtered = df.copy()
        strict_count = len(df)
        filter_messages = []

        if filters.get('price_range'):
            min_price, max_price = filters['price_range']

            if relax_level == 1:
                if max_price != float('inf'):
                    max_price *= 1.2
                if min_price > 0:
                    min_price *= 0.8
                filter_messages.append(f"💰 Price filter relaxed 20%: Rp {min_price:,.0f} - Rp {max_price:,.0f}")
            elif relax_level == 2:
                if max_price != float('inf'):
                    max_price *= 1.4
                if min_price > 0:
                    min_price *= 0.6
                filter_messages.append(f"💰 Price filter relaxed 40%: Rp {min_price:,.0f} - Rp {max_price:,.0f}")
            else:
                filter_messages.append(f"✅ Price filter (strict): Rp {min_price:,.0f} - Rp {max_price:,.0f}")

            filtered = filtered[
                (filtered['price'] >= min_price) &
                (filtered['price'] <= max_price)
                ]

        if filters.get('min_rating'):
            min_rating = filters['min_rating']

            if relax_level == 1:
                min_rating -= 0.3
                filter_messages.append(f"⭐ Rating filter relaxed: ≥ {min_rating:.1f}")
            elif relax_level == 2:
                min_rating -= 0.5
                filter_messages.append(f"⭐ Rating filter relaxed: ≥ {min_rating:.1f}")
            else:
                filter_messages.append(f"✅ Rating filter (strict): ≥ {min_rating}")

            filtered = filtered[filtered['rating'] >= min_rating]

        if filters.get('size_range'):
            min_size, max_size = filters['size_range']

            if relax_level >= 1:
                if max_size != float('inf'):
                    max_size *= 1.5
                if min_size > 0:
                    min_size *= 0.5
                filter_messages.append(f"📏 Size filter relaxed: {min_size:.0f} - {max_size:.0f} ml")
            else:
                filter_messages.append(f"✅ Size filter (strict): {min_size:.0f} - {max_size:.0f} ml")

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
            filter_messages.append(f"✅ Ingredient filter: {', '.join(ingredients)}")

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

        # Parse preferences
        filters = {}
        if preferences:
            self.filter_logs.append("🔍 Parsing preferences...")

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
                self.filter_logs.append(f"✅ Detected {len(filters)} filter(s)")

        # Adaptive filter application
        working_df = self.products.copy()
        coverage_metrics = None

        if filters:
            self.filter_logs.append("🔄 Applying filters with adaptive relaxation...")

            # Try strict first
            working_df, coverage_metrics = self.apply_filters(self.products.copy(), filters, relax_level=0)
            self.filter_logs.append(f"📊 Strict filter results: {len(working_df)} products")

            # Relax 20% if needed
            if len(working_df) < top_n:
                self.filter_logs.append(f"⚠️ Insufficient results ({len(working_df)} < {top_n})")
                self.filter_logs.append(f"🔄 Relaxing filters by 20%...")
                working_df, coverage_metrics = self.apply_filters(self.products.copy(), filters, relax_level=1)
                self.filter_logs.append(f"📊 Relaxed filter results: {len(working_df)} products")

            # Relax 40% if still needed
            if len(working_df) < top_n:
                self.filter_logs.append(f"⚠️ Still insufficient ({len(working_df)} < {top_n})")
                self.filter_logs.append(f"🔄 Relaxing filters by 40%...")
                working_df, coverage_metrics = self.apply_filters(self.products.copy(), filters, relax_level=2)
                self.filter_logs.append(f"📊 Final results: {len(working_df)} products")

            if working_df.empty:
                self.filter_logs.append("❌ No products match even with relaxed filters")
                return pd.DataFrame()

        # Input expansion and vectorization
        expanded_problems = InputExpander.expand_problems(problems)
        user_input = expanded_problems.copy()
        if categories:
            user_input.extend(categories)
        if preferences:
            user_input.append(preferences)

        user_vector = self.vectorizer.vectorize(user_input)

        # Get filtered indices
        filtered_indices = working_df.index.tolist()
        filtered_vectors = self.product_vectors[filtered_indices]
        similarities = cosine_similarity([user_vector], filtered_vectors)[0]

        # Category boost
        if categories:
            for i, idx in enumerate(filtered_indices):
                if working_df.loc[idx, 'category'] in categories:
                    similarities[i] = min(similarities[i] + 0.05, 1.0)

        # Distribution logic
        if categories:
            results = []
            per_cat_base = top_n // len(categories)
            remainder = top_n % len(categories)

            category_pools = {}
            for cat in categories:
                cat_mask = working_df['category'] == cat
                cat_local_indices = [i for i, idx in enumerate(filtered_indices) if cat_mask[idx]]

                if cat_local_indices:
                    cat_items = [(filtered_indices[i], similarities[i]) for i in cat_local_indices]
                    cat_items.sort(key=lambda x: x[1], reverse=True)
                    category_pools[cat] = cat_items
                else:
                    category_pools[cat] = []

            cat_pointers = {cat: 0 for cat in categories}

            for round_num in range(per_cat_base):
                for cat in categories:
                    if cat_pointers[cat] < len(category_pools[cat]):
                        idx, sim = category_pools[cat][cat_pointers[cat]]
                        results.append({
                            'idx': int(idx),
                            'similarity': float(sim),
                            'is_strict_match': coverage_metrics['relax_level'] == 0 if coverage_metrics else True
                        })
                        cat_pointers[cat] += 1

            for i in range(remainder):
                cat = categories[i]
                if cat_pointers[cat] < len(category_pools[cat]):
                    idx, sim = category_pools[cat][cat_pointers[cat]]
                    results.append({
                        'idx': int(idx),
                        'similarity': float(sim),
                        'is_strict_match': coverage_metrics['relax_level'] == 0 if coverage_metrics else True
                    })
                    cat_pointers[cat] += 1
        else:
            top_local_indices = similarities.argsort()[-top_n:][::-1]
            results = [
                {
                    'idx': int(filtered_indices[i]),
                    'similarity': float(similarities[i]),
                    'is_strict_match': coverage_metrics['relax_level'] == 0 if coverage_metrics else True
                }
                for i in top_local_indices
            ]

        results.sort(key=lambda x: x['similarity'], reverse=True)

        if coverage_metrics:
            for item in results:
                item['relax_level'] = coverage_metrics['relax_level']

        return self._format_results_from_indices(results, problems, coverage_metrics)

    def _format_results_from_indices(self, results, problems, coverage_metrics):
        formatted = []
        for item in results:
            idx = item['idx']
            product = self.products.iloc[idx]

            desc = product.get('claim', '')
            if pd.isna(desc) or not desc:
                desc = str(product.get('text_embedding', ''))[:150] + '...'

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

#Navigation
selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Rekomendasi", "About"],
    icons=["house-fill", "search-heart", "info-circle-fill"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)


def show_alert(message, alert_type="success"):
    """Display styled alert"""
    alert_class = f"alert-{alert_type}"
    st.markdown(f'<div class="{alert_class}">🎉 {message}</div>', unsafe_allow_html=True)

def save_recommendations_to_csv(df, user_name):
    """Save recommendations to CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recommendations_{user_name.replace(' ', '_')}_{timestamp}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    return filename


# PAGE 1: DASHBOARD
if selected == "Dashboard":
    st.markdown('<h1 style="text-align:center;color:white;">💇‍♀️ HairCare Recommender System</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:white;font-size:1.2rem;margin-bottom:3rem;">Sistem Rekomendasi Produk Perawatan Rambut dengan Adaptive Filtering</p>',
        unsafe_allow_html=True)

    # Stats
    if recommender:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(recommender.products)}</div>
                <div class="stats-label">Total Produk</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(recommender.products['brand'].unique())}</div>
                <div class="stats-label">Total Brand</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(recommender.products['category'].unique())}</div>
                <div class="stats-label">Kategori</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">AI</div>
                <div class="stats-label">Powered</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Features
    st.markdown('<h2 style="color:white;text-align:center;margin-bottom:2rem;">✨ Kenali Masalah Rambutmu</h2>',
                unsafe_allow_html=True)

    # Hair Problems Info Cards
    problems = [
        {
            "title": "🏜️ Rambut Kering & Kusam",
            "desc": "Rambut kering terjadi ketika rambut kehilangan kelembaban alami. Penyebabnya bisa dari paparan sinar matahari, penggunaan alat styling panas berlebihan, atau kurangnya perawatan yang tepat. Solusinya adalah menggunakan hair oil atau hair mask yang kaya nutrisi untuk mengembalikan kelembaban rambut."
        },
        {
            "title": "💧 Rambut Berminyak & Lepek",
            "desc": "Produksi sebum berlebih di kulit kepala menyebabkan rambut cepat berminyak dan lepek. Kondisi ini bisa dipicu oleh hormon, cuaca panas, atau penggunaan produk yang tidak cocok. Gunakan shampoo khusus untuk rambut berminyak dengan formula yang ringan namun efektif membersihkan."
        },
        {
            "title": "⚠️ Rambut Rontok",
            "desc": "Kerontokan rambut normal terjadi 50-100 helai per hari. Namun jika berlebihan, bisa disebabkan oleh stress, kekurangan nutrisi, atau masalah hormon. Treatment intensif dengan serum anti-rontok dan vitamin rambut dapat membantu memperkuat akar rambut dan mengurangi kerontokan."
        },
        {
            "title": "🦠 Ketombe & Kulit Kepala Gatal",
            "desc": "Ketombe disebabkan oleh pertumbuhan jamur Malassezia atau kulit kepala yang terlalu kering/berminyak. Gejalanya berupa serpihan putih dan rasa gatal. Shampoo anti-ketombe dengan kandungan zinc pyrithione atau ketoconazole efektif mengatasi masalah ini."
        },
        {
            "title": "✂️ Ujung Rambut Bercabang",
            "desc": "Split ends atau ujung rambut bercabang terjadi ketika lapisan pelindung rambut (cuticle) rusak. Penyebabnya termasuk penggunaan alat panas, pewarnaan berlebihan, atau jarang memotong rambut. Gunakan hair serum dan conditioner untuk melindungi dan menutrisi ujung rambut."
        }
    ]

    # Display problem cards in single row with 5 columns
    cols = st.columns(5, gap="medium")

    for i, problem in enumerate(problems):
        with cols[i]:
            st.markdown(f"""
                <div class="stats-card" style="text-align:left;height:100%;min-height:280px;">
                    <h3 style="color:#667eea;margin-bottom:1rem;font-size:1.1rem;">{problem['title']}</h3>
                    <p style="color:#666;line-height:1.6;font-size:0.9rem;">{problem['desc']}</p>
                </div>
                """, unsafe_allow_html=True)

# REKOMENDASI
elif selected == "Rekomendasi":
    st.markdown('<h1 style="color:white;text-align:center;">🔍 Dapatkan Rekomendasi Produk</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:white;text-align:center;font-size:1.1rem;margin-bottom:2rem;">Lengkapi form di bawah untuk mendapatkan rekomendasi produk haircare yang sesuai</p>',
        unsafe_allow_html=True)

    if recommender is None:
        st.error("❌ Backend recommender gagal dimuat. Periksa path file!")
        st.stop()

    # Input Form
    with st.container():
        st.markdown(
            '<div style="background:white;padding:2rem;border-radius:15px;box-shadow:0 5px 20px rgba(0,0,0,0.2);">',
            unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("### 👤 Informasi Dasar")
            user_name = st.text_input(
                "Nama Anda",
                placeholder="Masukkan nama Anda",
                help="Nama akan digunakan untuk menyimpan hasil rekomendasi"
            )

            st.markdown("### 💇 Masalah Rambut")
            st.markdown(
                '<p style="font-size:0.9rem;color:#666;margin-bottom:0.5rem;">Pilih semua masalah yang sedang dialami (minimal 1)</p>',
                unsafe_allow_html=True)

            hair_problems = st.multiselect(
                "Pilih masalah rambut",
                ["ketombe", "rontok", "kering", "berminyak", "bercabang"],
                help="Pilih satu atau lebih masalah rambut",
                label_visibility="collapsed"
            )

            st.markdown("### 🛍️ Kategori Produk (Opsional)")
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
            st.markdown("### 💡 Preferensi (Opsional)")
            st.markdown(
                '<p style="font-size:0.9rem;color:#666;margin-bottom:0.5rem;">Masukkan preferensi untuk filtering lebih detail</p>',
                unsafe_allow_html=True)

            preferences = st.text_area(
                "Preferensi Anda",
                placeholder="Contoh: murah, rating tinggi, argan oil, dibawah 100rb, natural",
                height=100,
                help="Kombinasi bebas: harga, rating, ingredients, ukuran",
                label_visibility="collapsed"
            )

            # Preference hints
            st.markdown("""
            <div class="pref-hint">
                <strong>💰 Harga:</strong> <span class="pref-example">"murah", "dibawah 50rb", "50-100rb", ">200rb"</span><br>
                <strong>⭐ Rating:</strong> <span class="pref-example">"rating tinggi", "bintang 4", "4.5 keatas"</span><br>
                <strong>🧪 Ingredients:</strong> <span class="pref-example">"argan oil", "sulfate free", "natural"</span><br>
                <strong>📏 Ukuran:</strong> <span class="pref-example">"ukuran besar", "diatas 200ml", "100ml"</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🔢 Jumlah Rekomendasi")
            top_n = st.slider("Berapa produk yang ingin direkomendasi?", 5, 10, 10)

        st.markdown('</div>', unsafe_allow_html=True)

    # Submit Button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_btn = st.button("✨ Dapatkan Rekomendasi", use_container_width=True, type="primary")

    # Process Recommendation
    if submit_btn:
        if not user_name or not user_name.strip():
            show_alert("⚠️ Mohon masukkan nama Anda", "warning")
        elif not hair_problems:
            show_alert("⚠️ Mohon pilih minimal 1 masalah rambut", "warning")
        else:
            with st.spinner("🔄 Memproses rekomendasi dengan adaptive filtering..."):
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
                st.session_state.filter_logs = recommender.filter_logs
                st.session_state.show_detail = False

    # Display Results
    if hasattr(st.session_state, 'recommendations') and not st.session_state.recommendations.empty:
        df_recs = st.session_state.recommendations

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("---")

        # Success Alert
        show_alert(f"Berhasil menemukan {len(df_recs)} rekomendasi produk untuk {st.session_state.user_name}!",
                   "success")

        # Filter Logs
        if hasattr(st.session_state, 'filter_logs') and st.session_state.filter_logs:
            with st.expander("📋 Lihat Log Filter Process", expanded=False):
                for log in st.session_state.filter_logs:
                    st.markdown(f"- {log}")

        # Statistics
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

        # Coverage Metrics
        if hasattr(df_recs, 'attrs') and 'coverage_metrics' in df_recs.attrs:
            metrics = df_recs.attrs['coverage_metrics']
            if metrics['relax_level'] > 0:
                strict_count = sum(df_recs['is_strict_match'])
                total_count = len(df_recs)
                coverage = (strict_count / total_count * 100) if total_count > 0 else 0

                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"""
                    <div class="stats-card" style="text-align:left;">
                        <h3 style="color:#667eea;">📊 Filter Coverage</h3>
                        <p style="font-size:1.1rem;margin-top:0.5rem;">
                        <span class="filter-badge badge-strict">{strict_count} Strict Matches</span>
                        <span class="filter-badge badge-relaxed-{metrics['relax_level']}">{total_count - strict_count} Relaxed Matches</span>
                        </p>
                        <p style="color:#666;margin-top:1rem;">Coverage: <strong>{coverage:.1f}%</strong> strict matches</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="stats-card" style="text-align:left;">
                        <h3 style="color:#667eea;">🔧 Relax Level</h3>
                        <p style="font-size:1.1rem;margin-top:0.5rem;">
                        <span class="filter-badge badge-relaxed-{metrics['relax_level']}">Level {metrics['relax_level']}</span>
                        </p>
                        <p style="color:#666;margin-top:1rem;">Filter relaxed by <strong>{metrics['relax_level'] * 20}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)

        # Save Button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("💾 Simpan Hasil ke CSV", use_container_width=True, type="secondary"):
                try:
                    filename = save_recommendations_to_csv(df_recs, st.session_state.user_name)
                    show_alert(f"✅ Rekomendasi berhasil disimpan ke: {filename}", "success")

                    # Provide download button
                    with open(filename, 'rb') as f:
                        st.download_button(
                            label="📥 Download File CSV",
                            data=f,
                            file_name=filename,
                            mime='text/csv',
                            use_container_width=True
                        )
                except Exception as e:
                    show_alert(f"❌ Gagal menyimpan file: {str(e)}", "warning")

        # Product Cards
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 🎯 Hasil Rekomendasi")
        st.markdown(
            f"<p style='color:white;'>Masalah: <strong>{', '.join(st.session_state.hair_problems)}</strong></p>",
            unsafe_allow_html=True)

        # Display in 5 columns
        cols = st.columns(5, gap="medium")

        for i, (idx, row) in enumerate(df_recs.iterrows()):
            col = cols[i % 5]

            with col:
                # Product card with relaxed indicator
                card_class = "product-card-small relaxed" if row.get('relax_level', 0) > 0 else "product-card-small"

                with st.container():
                    # Image
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

                    # Info
                    st.markdown(f"**{row['name'][:40]}...**" if len(row['name']) > 40 else f"**{row['name']}**")
                    st.markdown(f"<p style='font-size:0.85rem;color:#999;'>{row['brand']}</p>", unsafe_allow_html=True)

                    try:
                        price_str = f"Rp {int(float(row['price'])):,}"
                    except:
                        price_str = str(row['price'])

                    st.markdown(f"<p style='font-size:0.9rem;font-weight:700;color:#f5576c;'>{price_str}</p>",
                                unsafe_allow_html=True)
                    st.markdown(f"⭐ {row['rating']:.1f}/5 ({int(row['total_reviews'])} review)")

                    # Match badge
                    if row.get('relax_level', 0) > 0:
                        st.markdown(
                            f'<span class="filter-badge badge-relaxed-{row["relax_level"]}">{row["similarity"]:.0%} match (Relaxed)</span>',
                            unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="filter-badge badge-strict">{row["similarity"]:.0%} match</span>',
                                    unsafe_allow_html=True)

                    # Detail button
                    if st.button(f"📖 Detail", key=f"detail_{i}", use_container_width=True):
                        st.session_state.selected_product_idx = i
                        st.session_state.show_detail = True
                        st.rerun()

        # Detail View
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

                # Relax indicator
                if row.get('relax_level', 0) > 0:
                    st.markdown(
                        f'<span class="filter-badge badge-relaxed-{row["relax_level"]}">Filter Relaxed Level {row["relax_level"]}</span>',
                        unsafe_allow_html=True)
                else:
                    st.markdown('<span class="filter-badge badge-strict">Strict Match ✓</span>', unsafe_allow_html=True)

                if row.get('product_url') and str(row.get('product_url')).strip() and str(
                        row.get('product_url')) != 'nan':
                    st.markdown(f"[🛒 **Beli di Merchant**]({row.get('product_url')})")

            st.markdown("---")

            tab1, tab2, tab3 = st.tabs(["📝 Deskripsi", "📖 Cara Pakai", "🧪 Ingredients"])

            with tab1:
                st.write(row.get('description', '-') if row.get('description') else '-')

            with tab2:
                st.write(row.get('how_to_use', '-') if row.get('how_to_use') else '-')

            with tab3:
                st.write(row.get('ingredients', '-') if row.get('ingredients') else '-')

# ABOUT
elif selected == "About":
    st.markdown('<h1 style="color:white;text-align:center;">ℹ️ Tentang Sistem</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:white;text-align:center;font-size:1.1rem;margin-bottom:2rem;">HairCare Recommender dengan Adaptive Filtering Technology</p>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-card" style="text-align:left;">
        <h2 style="color:#667eea;">🚀 Teknologi yang Digunakan</h2>
        <ul style="line-height:2;color:#666;font-size:1rem;">
            <li><strong>TF-IDF Vectorization:</strong> Ekstraksi fitur dari teks produk</li>
            <li><strong>FastText Embeddings (Fine-tuned):</strong> Word embeddings khusus domain haircare</li>
            <li><strong>Cosine Similarity:</strong> Perhitungan kemiripan produk dengan kebutuhan user</li>
            <li><strong>Adaptive Filtering:</strong> Sistem filter cerdas dengan auto-relaxation</li>
            <li><strong>Typo Correction:</strong> Koreksi otomatis kesalahan pengetikan</li>
            <li><strong>Input Expansion:</strong> Ekspansi query untuk hasil lebih akurat</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="stats-card" style="text-align:left;">
        <h2 style="color:#667eea;">📋 Cara Penggunaan</h2>
        <ol style="line-height:2;color:#666;font-size:1rem;">
            <li><strong>Input Nama:</strong> Masukkan nama Anda untuk personalisasi</li>
            <li><strong>Pilih Masalah:</strong> Pilih 1 atau lebih masalah rambut yang dialami</li>
            <li><strong>Pilih Kategori:</strong> (Opsional) Pilih jenis produk yang diinginkan</li>
            <li><strong>Set Preferensi:</strong> (Opsional) Tambahkan filter harga, rating, ingredients, ukuran</li>
            <li><strong>Dapatkan Rekomendasi:</strong> Sistem akan memberikan top-N produk terbaik</li>
            <li><strong>Simpan Hasil:</strong> Export hasil ke CSV untuk dokumentasi</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="stats-card" style="text-align:left;">
            <h3 style="color:#667eea;">✨ Keunggulan</h3>
            <ul style="line-height:1.8;color:#666;">
                <li>Filter adaptif otomatis</li>
                <li>Distribusi kategori merata</li>
                <li>Parse preferensi natural language</li>
                <li>Typo correction built-in</li>
                <li>Export hasil dengan metadata</li>
                <li>Coverage metrics tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="stats-card" style="text-align:left;">
            <h3 style="color:#667eea;">🎯 Contoh Preferensi</h3>
            <ul style="line-height:1.8;color:#666;">
                <li>"murah rating tinggi"</li>
                <li>"dibawah 100rb natural"</li>
                <li>"50-150rb argan oil"</li>
                <li>"premium rating 4.5 keatas"</li>
                <li>"ukuran besar sulfate free"</li>
                <li>"200ml diatas rating bagus"</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:white;padding:2rem;">
        <p style="font-size:1.2rem;margin-bottom:0.5rem;">💇‍♀️ HairCare Recommender System</p>
        <p style="font-size:1rem;">Powered by FastText + TF-IDF | Adaptive Filtering Technology</p>
        <p style="font-size:0.9rem;margin-top:1rem;opacity:0.8;">© 2025 | Made with ❤️ for better hair care</p>
    </div>
    """, unsafe_allow_html=True)