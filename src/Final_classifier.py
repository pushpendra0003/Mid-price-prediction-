import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Conv1D, MaxPooling1D, Flatten, 
    Concatenate, Dropout, BatchNormalization, Bidirectional,
    LayerNormalization, MultiHeadAttention, Reshape, Add,
    GlobalAveragePooling1D, GRU, Activation, Multiply, Lambda
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, CSVLogger
)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os
import json
import time
import logging
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure GPU memory growth and mixed precision
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        # Set memory limit to 90% of available GPU memory
        tf.config.set_logical_device_configuration(
            device,
            [tf.config.LogicalDeviceConfiguration(memory_limit=24564 * 0.9)]  # 90% of 24GB
        )
    logger.info(f"Found {len(physical_devices)} GPU(s): {physical_devices}")
else:
    logger.info("No GPU found. Running on CPU.")

# Set mixed precision policy for better GPU utilization
tf.keras.mixed_precision.set_global_policy('float32')
logger.info("Mixed precision policy set to: float32")

# Enable XLA compilation for better performance
tf.config.optimizer.set_jit(True)

# Configure for better GPU utilization
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Set environment variables for better GPU utilization
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '2'
os.environ['TF_USE_CUDNN'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def parse_orderbook_data(df):
    """Parse order book columns from the raw data."""
    orderbook_cols = []
    for i in range(1, 11):
        bid_price_col = df.columns[2 + (i-1)*2]  # Bid price columns
        bid_qty_col = df.columns[3 + (i-1)*2]    # Bid quantity columns
        ask_price_col = df.columns[22 + (i-1)*2] # Ask price columns
        ask_qty_col = df.columns[23 + (i-1)*2]   # Ask quantity columns
        
        # Rename columns to standard format
        df = df.rename(columns={
            bid_price_col: f'BID_{i}_PRICE',
            bid_qty_col: f'BID_{i}_QUANTITY',
            ask_price_col: f'ASK_{i}_PRICE',
            ask_qty_col: f'ASK_{i}_QUANTITY'
        })
        
        orderbook_cols.extend([
            f'BID_{i}_PRICE', f'BID_{i}_QUANTITY',
            f'ASK_{i}_PRICE', f'ASK_{i}_QUANTITY'
        ])
    
    return df, orderbook_cols

class EnhancedFeatureManager:
    """Enhanced feature management with advanced feature engineering."""
    
    def __init__(self, df):
        self.df = df
        self.all_columns = df.columns.tolist()
        self.feature_groups = {}
        self.scalers = {}
        self._analyze_columns()
        self._engineer_features()
    
    def _analyze_columns(self):
        """Analyze and categorize columns."""
        # Order book features (bid-ask pairs)
        orderbook_features = []
        for i in range(1, 11):
            bid_price = f'BID_{i}_PRICE'
            bid_qty = f'BID_{i}_QUANTITY'
            ask_price = f'ASK_{i}_PRICE'
            ask_qty = f'ASK_{i}_QUANTITY'
            
            if all(col in self.all_columns for col in [bid_price, bid_qty, ask_price, ask_qty]):
                orderbook_features.extend([bid_price, bid_qty, ask_price, ask_qty])
        
        if orderbook_features:
            self.feature_groups['orderbook'] = orderbook_features
        
        # Historical returns features
        returns_features = [
            col for col in self.all_columns 
            if 'HISTORICAL_RETURNS' in col.upper()
        ]
        if returns_features:
            self.feature_groups['returns'] = returns_features
        
        # Technical features
        tech_features = [
            col for col in self.all_columns 
            if any(indicator in col.upper() for indicator in ['EMA', 'SMA', 'RSI', 'MACD', 'VWAP'])
        ]
        if tech_features:
            self.feature_groups['technical'] = tech_features
        
        # Microstructure features
        micro_features = [
            col for col in self.all_columns 
            if any(term in col.upper() for term in ['VOLUME', 'IMBALANCE', 'PRESSURE', 'SPREAD'])
        ]
        if micro_features:
            self.feature_groups['microstructure'] = micro_features
    
    def _engineer_features(self):
        """Engineer additional features."""
        df = self.df
        
        # Add timestamp features
        df['hour'] = pd.to_datetime(df.index).hour
        df['minute'] = pd.to_datetime(df.index).minute
        df['time_of_day'] = df['hour'] + df['minute'] / 60
        df['time_cos'] = np.cos(2 * np.pi * df['time_of_day'] / 24)
        df['time_sin'] = np.sin(2 * np.pi * df['time_of_day'] / 24)
        self.feature_groups.setdefault('time', []).extend(['time_cos', 'time_sin'])
        
        # Volatility features
        if 'MID_PRICE' in df.columns:
            df['volatility_5min'] = df['MID_PRICE'].pct_change().rolling(5).std()
            df['volatility_15min'] = df['MID_PRICE'].pct_change().rolling(15).std()
            df['volatility_ratio'] = df['volatility_5min'] / df['volatility_15min']
            self.feature_groups.setdefault('technical', []).extend([
                'volatility_5min', 'volatility_15min', 'volatility_ratio'
            ])
        
        # Adaptive features based on volatility regime
        if 'volatility_15min' in df.columns:
            vol_median = df['volatility_15min'].median()
            df['high_vol_regime'] = (df['volatility_15min'] > vol_median).astype(float)
            self.feature_groups.setdefault('technical', []).append('high_vol_regime')
        
        # Original features
        orderbook_features = self.feature_groups['orderbook']
        returns_features = self.feature_groups['returns']
        tech_features = self.feature_groups['technical']
        micro_features = self.feature_groups['microstructure']
        
        # Price spread features
        if 'BID_1_PRICE' in self.all_columns and 'ASK_1_PRICE' in self.all_columns:
            df['SPREAD'] = df['ASK_1_PRICE'] - df['BID_1_PRICE']
            df['RELATIVE_SPREAD'] = df['SPREAD'] / df['BID_1_PRICE']
            tech_features.extend(['SPREAD', 'RELATIVE_SPREAD'])
        
        # Volume imbalance
        if 'BID_1_QUANTITY' in self.all_columns and 'ASK_1_QUANTITY' in self.all_columns:
            df['VOLUME_IMBALANCE'] = (
                df['BID_1_QUANTITY'] - df['ASK_1_QUANTITY']
            ) / (df['BID_1_QUANTITY'] + df['ASK_1_QUANTITY'])
            micro_features.append('VOLUME_IMBALANCE')
        
        # Price pressure
        for i in range(1, 5):
            bid_curr = f'BID_{i}_PRICE'
            bid_next = f'BID_{i+1}_PRICE'
            ask_curr = f'ASK_{i}_PRICE'
            ask_next = f'ASK_{i+1}_PRICE'
            
            if all(col in self.all_columns for col in [bid_curr, bid_next]):
                df[f'BID_PRESSURE_{i}'] = df[bid_curr] - df[bid_next]
                micro_features.append(f'BID_PRESSURE_{i}')
            
            if all(col in self.all_columns for col in [ask_curr, ask_next]):
                df[f'ASK_PRESSURE_{i}'] = df[ask_next] - df[ask_curr]
                micro_features.append(f'ASK_PRESSURE_{i}')
        
        # Current mid price features
        if 'BID_1_PRICE' in self.all_columns and 'ASK_1_PRICE' in self.all_columns:
            df['MID_PRICE'] = (df['BID_1_PRICE'] + df['ASK_1_PRICE']) / 2
            df['MID_PRICE_RETURNS'] = df['MID_PRICE'].pct_change()
            returns_features.append('MID_PRICE_RETURNS')
        
        # Order book imbalance
        bid_qty_cols = [f'BID_{i}_QUANTITY' for i in range(1, 6)]
        ask_qty_cols = [f'ASK_{i}_QUANTITY' for i in range(1, 6)]
        
        if all(col in self.all_columns for col in bid_qty_cols + ask_qty_cols):
            bid_quantities = df[bid_qty_cols]
            ask_quantities = df[ask_qty_cols]
            df['ORDER_BOOK_IMBALANCE'] = (
                bid_quantities.sum(axis=1) - ask_quantities.sum(axis=1)
            ) / (bid_quantities.sum(axis=1) + ask_quantities.sum(axis=1))
            micro_features.append('ORDER_BOOK_IMBALANCE')
        
        # Fill NaN values
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].fillna(method='ffill').fillna(0)
        
        self.df = df
    
    def prepare_features(self, lookback=20):
        """Prepare features for 30-minute horizon."""
        sequences = {}
        labels = None
        
        # Prepare feature sequences
        for group, features in self.feature_groups.items():
            valid_features = [f for f in features if f in self.df.columns]
            if not valid_features:
                continue
            
            data = self.df[valid_features].values
            
            if group == 'returns':
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            scaled_data = scaler.fit_transform(data)
            self.scalers[group] = scaler
            sequences[group] = self._create_sequences(scaled_data, lookback)
        
        # Prepare labels for 30-minute horizon
        if 'FUTURE_RETURNS_30min' in self.df.columns:
            labels = (self.df['FUTURE_RETURNS_30min'].values > 0).astype(int)
            labels = labels[lookback:]
        
        return sequences, labels
    
    def _create_sequences(self, data, lookback):
        """Create sequences for time series data."""
        sequences = []
        for i in range(len(data) - lookback):
            sequences.append(data[i:(i + lookback)])
        return np.array(sequences)

class EnhancedDataGenerator(tf.keras.utils.Sequence):
    """Enhanced data generator with better memory management and augmentation."""
    
    def __init__(self, X_dict, y, batch_size=32, augment=False):
        self.X_dict = X_dict
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.y))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.y) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get batch for each feature group
        X_batch = {
            group: self.X_dict[group][batch_indices] 
            for group in self.X_dict.keys()
        }
        y_batch = self.y[batch_indices]
        
        if self.augment:
            X_batch, y_batch = self._augment_batch(X_batch, y_batch)
        
        # Return as list in the same order as model expects
        return [X_batch[group] for group in self.X_dict.keys()], y_batch
    
    def _augment_batch(self, X_batch, y_batch):
        """Apply data augmentation techniques."""
        # Add random noise
        for group in X_batch.keys():
            noise = np.random.normal(0, 0.01, X_batch[group].shape)
            X_batch[group] = X_batch[group] + noise
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def compute_class_weights(y):
    """Compute balanced class weights."""
    total = len(y)
    class_counts = np.bincount(y)
    return {
        i: total / (len(class_counts) * count)
        for i, count in enumerate(class_counts)
    }

@tf.keras.saving.register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.false_positives = self.add_weight(name='fp', initializer='zeros', dtype=tf.float32)
        self.false_negatives = self.add_weight(name='fn', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert inputs to float32
        y_pred = tf.cast(tf.round(y_pred), tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # Calculate metrics with explicit float32 casting
        true_positives = tf.cast(tf.reduce_sum(y_true * y_pred), tf.float32)
        false_positives = tf.cast(tf.reduce_sum((1 - y_true) * y_pred), tf.float32)
        false_negatives = tf.cast(tf.reduce_sum(y_true * (1 - y_pred)), tf.float32)
        
        # Update state
        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        # Ensure all calculations are in float32
        precision = tf.cast(self.true_positives, tf.float32) / (
            tf.cast(self.true_positives, tf.float32) + 
            tf.cast(self.false_positives, tf.float32) + 
            tf.keras.backend.epsilon()
        )
        recall = tf.cast(self.true_positives, tf.float32) / (
            tf.cast(self.true_positives, tf.float32) + 
            tf.cast(self.false_negatives, tf.float32) + 
            tf.keras.backend.epsilon()
        )
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

class ConfidenceScore(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.2, name='confidence_score', **kwargs):
        super(ConfidenceScore, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.confident_predictions = self.add_weight(name='confident_preds', initializer='zeros')
        self.total_predictions = self.add_weight(name='total_preds', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate distance from decision boundary (0.5)
        confidence_margins = tf.abs(y_pred - 0.5)
        # Count predictions with high confidence
        confident_count = tf.reduce_sum(tf.cast(confidence_margins > self.threshold, tf.float32))
        total_count = tf.cast(tf.size(y_pred), tf.float32)
        
        self.confident_predictions.assign_add(confident_count)
        self.total_predictions.assign_add(total_count)

    def result(self):
        return self.confident_predictions / (self.total_predictions + tf.keras.backend.epsilon())

    def reset_state(self):
        self.confident_predictions.assign(0)
        self.total_predictions.assign(0)

def create_enhanced_model(feature_shapes, dropout_rate=0.3):
    """Create an enhanced model with state-of-the-art architecture."""
    # Set mixed precision policy
    policy = tf.keras.mixed_precision.Policy('float32')  # Change to float32 for consistency
    tf.keras.mixed_precision.set_global_policy(policy)
    
    inputs = {}
    processed_features = {}
    
    # Process each feature group with advanced architectures
    for group, shape in feature_shapes.items():
        inputs[group] = Input(shape=shape, name=f'{group}_input', dtype='float32')
        x = inputs[group]
        
        # Enhanced positional encoding
        if len(shape) > 1:
            # Create time features with correct shape and type
            time_features = tf.range(shape[0], dtype=tf.float32)
            time_features = tf.expand_dims(time_features, axis=0)
            time_features = tf.expand_dims(time_features, axis=-1)
            
            # Create sinusoidal encoding with proper type
            sin_features = tf.sin(time_features / 10000.0)
            cos_features = tf.cos(time_features / 10000.0)
            
            # Concatenate time features with proper type casting
            time_encoding = tf.concat([sin_features, cos_features], axis=-1)
            time_encoding = tf.cast(time_encoding, tf.float32)
            
            if len(x.shape) == 3:
                time_encoding = tf.tile(time_encoding, [tf.shape(x)[0], 1, 1])
                x = tf.concat([x, time_encoding], axis=-1)
            else:
                time_encoding = tf.squeeze(time_encoding, axis=0)
                x = tf.concat([x, time_encoding], axis=-1)
        
        # Group-specific processing with proper type handling
        if group in ['orderbook', 'microstructure']:
            x = Conv1D(128, 3, padding='same', dtype='float32')(x)
            x = BatchNormalization(dtype='float32')(x)
            x = Activation('swish', dtype='float32')(x)
            
            se = GlobalAveragePooling1D(dtype='float32')(x)
            se = Dense(128 // 16, activation='swish', dtype='float32')(se)
            se = Dense(128, activation='sigmoid', dtype='float32')(se)
            se = Reshape((1, 128))(se)
            x = Multiply(dtype='float32')([x, se])
            
            for dilation_rate in [1, 2, 4, 8]:
                shortcut = x
                x = Conv1D(128, 3, padding='same', dilation_rate=dilation_rate, dtype='float32')(x)
                x = BatchNormalization(dtype='float32')(x)
                x = Activation('swish', dtype='float32')(x)
                x = Conv1D(128, 3, padding='same', dtype='float32')(x)
                x = BatchNormalization(dtype='float32')(x)
                x = Add(dtype='float32')([shortcut, x])
                x = Activation('swish', dtype='float32')(x)
            
        elif group in ['returns', 'technical']:
            x = Bidirectional(LSTM(64, return_sequences=True, dtype='float32'))(x)
            
            attention_output = MultiHeadAttention(
                num_heads=8,
                key_dim=32,
                dtype='float32'
            )(x, x, x, use_causal_mask=True)
            
            x = Add(dtype='float32')([x, attention_output])
            x = LayerNormalization(epsilon=1e-6, dtype='float32')(x)
            
            ffn = Dense(256, activation='swish', dtype='float32')(x)
            ffn = Dropout(dropout_rate)(ffn)
            ffn = Dense(128, dtype='float32')(ffn)
            
            x = Add(dtype='float32')([x, ffn])
            x = LayerNormalization(epsilon=1e-6, dtype='float32')(x)
        
        x = GlobalAveragePooling1D(dtype='float32')(x)
        x = Dense(128, activation='swish', dtype='float32')(x)
        x = Dropout(dropout_rate)(x)
        processed_features[group] = x
    
    if len(processed_features) > 1:
        combined = Concatenate(dtype='float32')(list(processed_features.values()))
        combined = Reshape((1, -1))(combined)
        attention = MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dtype='float32'
        )(combined, combined, combined)
        combined = Flatten()(attention)
        
        gate = Dense(combined.shape[-1], activation='sigmoid', dtype='float32')(combined)
        combined = Multiply(dtype='float32')([combined, gate])
    else:
        combined = list(processed_features.values())[0]
    
    x = combined
    for _ in range(3):
        h = Dense(256, activation='swish', dtype='float32')(x)
        t = Dense(256, activation='sigmoid', dtype='float32')(x)
        c = Dense(256, dtype='float32')(x)
        x = Add(dtype='float32')([
            Multiply(dtype='float32')([h, t]),
            Multiply(dtype='float32')([c, 1.0 - t])
        ])
        x = BatchNormalization(dtype='float32')(x)
        x = Dropout(dropout_rate)(x)
    
    shared = Dense(128, activation='swish', dtype='float32')(x)
    
    # Main prediction head with proper type casting
    prediction = Dense(1, activation='sigmoid', name='prediction', dtype='float32')(shared)
    
    # Confidence estimation head with proper type casting
    confidence_logits = Dense(1, name='confidence_logits', dtype='float32')(shared)
    temperature = tf.constant(2.0, dtype=tf.float32)
    confidence = Lambda(
        lambda x: tf.sigmoid(x / temperature),
        name='confidence',
        dtype='float32'
    )(confidence_logits)
    
    model = Model(
        inputs=list(inputs.values()),
        outputs=[prediction, confidence],
        name='enhanced_trading_model'
    )
    
    return model

def train_model(X_dict, y, n_splits=3, epochs=50, batch_size=1024):  # Increased batch size
    """Enhanced training function with advanced techniques."""
    
    # Initialize TimeSeriesSplit with gap
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=20)
    feature_shapes = {group: data.shape[1:] for group, data in X_dict.items()}
    
    # Initialize lists to store metrics
    all_histories = []
    all_val_predictions = []
    all_val_confidences = []
    all_val_true = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(y)):
        logger.info(f"\nTraining fold {fold + 1}/{n_splits}")
        
        # Split data
        X_train = {group: data[train_idx] for group, data in X_dict.items()}
        X_val = {group: data[val_idx] for group, data in X_dict.items()}
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Advanced class weighting
        class_weights = compute_class_weights(y_train)
        class_weights = {k: float(v) for k, v in class_weights.items()}
        
        # Create data generators with larger batch size
        train_gen = EnhancedDataGenerator(
            X_train,
            y_train,
            batch_size=batch_size,
            augment=True
        )
        val_gen = EnhancedDataGenerator(
            X_val,
            y_val,
            batch_size=batch_size,
            augment=False
        )
        
        # Create and compile model
        model = create_enhanced_model(feature_shapes)
        
        # Advanced learning rate schedule
        total_steps = epochs * len(train_gen)
        warmup_steps = total_steps // 10
        
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=0.001,
            first_decay_steps=total_steps // 3,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1
        )
        
        # Create optimizer with mixed precision
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            clipnorm=1.0,
            clipvalue=0.5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Advanced loss functions
        def prediction_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            return bce
        
        def confidence_loss(y_true, y_pred):
            # Ensure consistent types
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            # Round predictions to binary values
            y_true_rounded = tf.round(y_true)
            y_pred_rounded = tf.round(y_pred)
            
            # Calculate prediction correctness
            prediction_correct = tf.cast(
                tf.equal(y_true_rounded, y_pred_rounded),
                tf.float32
            )
            
            # Calculate MSE between correctness and confidence
            return tf.keras.losses.mse(prediction_correct, y_pred)
        
        # Compile model with mixed precision
        model.compile(
            optimizer=optimizer,
            loss={
                'prediction': prediction_loss,
                'confidence': confidence_loss
            },
            loss_weights={
                'prediction': 1.0,
                'confidence': 0.2
            },
            metrics={
                'prediction': [
                    'accuracy',
                    F1Score(),
                    'AUC',
                    'Precision',
                    'Recall',
                    ConfidenceScore()
                ],
                'confidence': ['mse', 'mae']
            }
        )
        
        # Advanced callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_prediction_loss',
                patience=10,
                restore_best_weights=True,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_prediction_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                mode='min'
            ),
            ModelCheckpoint(
                f'model_fold_{fold}.h5',
                monitor='val_prediction_loss',
                save_best_only=True,
                mode='min'
            ),
            CSVLogger(f'training_log_fold_{fold}.csv'),
            TensorBoard(
                log_dir=f'./logs/fold_{fold}',
                update_freq='epoch',
                profile_batch=0
            )
        ]
        
        # Train model with increased workers and use_multiprocessing
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            workers=8,  # Increased number of workers
            use_multiprocessing=True,
            max_queue_size=100  # Increased queue size
        )
        
        # Store results
        all_histories.append(history.history)
        predictions, confidences = model.predict(
            val_gen,
            batch_size=batch_size,
            workers=8,
            use_multiprocessing=True
        )
        all_val_predictions.extend(predictions)
        all_val_confidences.extend(confidences)
        all_val_true.extend(y_val)
        
        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()
    
    return all_histories, all_val_predictions, all_val_confidences, all_val_true

def plot_training_history(histories, save_path='training_results'):
    """Plot comprehensive training history across all folds."""
    os.makedirs(save_path, exist_ok=True)
    
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall', 'f1_score', 'confidence_score']
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 3)
    
    for idx, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        for fold, history in enumerate(histories):
            ax.plot(
                history[f'prediction_{metric}'],
                label=f'Train Fold {fold+1}',
                alpha=0.7
            )
            ax.plot(
                history[f'val_prediction_{metric}'],
                label=f'Val Fold {fold+1}',
                linestyle='--',
                alpha=0.7
            )
        
        ax.set_title(f'Model {metric.replace("_", " ").title()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_history.png')
    plt.close()
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    for fold, history in enumerate(histories):
        if 'val_confidence_mse' in history:
            plt.plot(
                history['val_confidence_mse'],
                label=f'Fold {fold+1}',
                alpha=0.7
            )
    plt.title('Validation Confidence MSE Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{save_path}/confidence_evolution.png')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, confidences, save_path='training_results'):
    """Plot enhanced confusion matrix with confidence analysis."""
    os.makedirs(save_path, exist_ok=True)
    
    # Convert predictions to binary and ensure 1D arrays
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    y_true = y_true.astype(int).flatten()
    confidences = confidences.flatten()
    
    # Create confidence thresholds
    high_conf_mask = confidences >= 0.8
    med_conf_mask = (confidences >= 0.6) & (confidences < 0.8)
    low_conf_mask = (confidences >= 0.4) & (confidences < 0.6)
    
    # Plot overall confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Overall Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{save_path}/confusion_matrix_overall.png')
    plt.close()
    
    # Plot confusion matrices by confidence level
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # High confidence
    if high_conf_mask.any():
        cm_high = confusion_matrix(
            y_true[high_conf_mask],
            y_pred_binary[high_conf_mask]
        )
        sns.heatmap(cm_high, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('High Confidence Predictions')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
    else:
        axes[0].text(0.5, 0.5, 'No high confidence predictions', 
                    ha='center', va='center')
        axes[0].set_title('High Confidence Predictions')
    
    # Medium confidence
    if med_conf_mask.any():
        cm_med = confusion_matrix(
            y_true[med_conf_mask],
            y_pred_binary[med_conf_mask]
        )
        sns.heatmap(cm_med, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_title('Medium Confidence Predictions')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
    else:
        axes[1].text(0.5, 0.5, 'No medium confidence predictions', 
                    ha='center', va='center')
        axes[1].set_title('Medium Confidence Predictions')
    
    # Low confidence
    if low_conf_mask.any():
        cm_low = confusion_matrix(
            y_true[low_conf_mask],
            y_pred_binary[low_conf_mask]
        )
        sns.heatmap(cm_low, annot=True, fmt='d', cmap='Blues', ax=axes[2])
        axes[2].set_title('Low Confidence Predictions')
        axes[2].set_ylabel('True Label')
        axes[2].set_xlabel('Predicted Label')
    else:
        axes[2].text(0.5, 0.5, 'No low confidence predictions', 
                    ha='center', va='center')
        axes[2].set_title('Low Confidence Predictions')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrix_by_confidence.png')
    plt.close()
    
    # Save metrics by confidence level
    metrics_by_confidence = {}
    
    if high_conf_mask.any():
        metrics_by_confidence['high_confidence'] = {
            'accuracy': accuracy_score(y_true[high_conf_mask], y_pred_binary[high_conf_mask]),
            'precision': precision_score(y_true[high_conf_mask], y_pred_binary[high_conf_mask]),
            'recall': recall_score(y_true[high_conf_mask], y_pred_binary[high_conf_mask]),
            'f1': f1_score(y_true[high_conf_mask], y_pred_binary[high_conf_mask]),
            'count': int(high_conf_mask.sum())
        }
    
    if med_conf_mask.any():
        metrics_by_confidence['medium_confidence'] = {
            'accuracy': accuracy_score(y_true[med_conf_mask], y_pred_binary[med_conf_mask]),
            'precision': precision_score(y_true[med_conf_mask], y_pred_binary[med_conf_mask]),
            'recall': recall_score(y_true[med_conf_mask], y_pred_binary[med_conf_mask]),
            'f1': f1_score(y_true[med_conf_mask], y_pred_binary[med_conf_mask]),
            'count': int(med_conf_mask.sum())
        }
    
    if low_conf_mask.any():
        metrics_by_confidence['low_confidence'] = {
            'accuracy': accuracy_score(y_true[low_conf_mask], y_pred_binary[low_conf_mask]),
            'precision': precision_score(y_true[low_conf_mask], y_pred_binary[low_conf_mask]),
            'recall': recall_score(y_true[low_conf_mask], y_pred_binary[low_conf_mask]),
            'f1': f1_score(y_true[low_conf_mask], y_pred_binary[low_conf_mask]),
            'count': int(low_conf_mask.sum())
        }
    
    with open(f'{save_path}/confidence_metrics.json', 'w') as f:
        json.dump(metrics_by_confidence, f, indent=4)

def visualize_saved_model_performance(model_path='best_model_30min.keras', save_path='model_visualization'):
    """Load and visualize the performance of a saved model."""
    os.makedirs(save_path, exist_ok=True)
    
    # Set mixed precision policy before loading model
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Load the saved model with custom objects
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            'F1Score': F1Score,
            'ConfidenceScore': ConfidenceScore
        },
        compile=False
    )
    
    # Recompile model with proper mixed precision settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0,
        clipvalue=0.5
    )
    
    model.compile(
        optimizer=optimizer,
        loss={
            'prediction': 'binary_crossentropy',
            'confidence': 'mse'
        },
        metrics={
            'prediction': [
                'accuracy',
                F1Score(),
                'AUC',
                'Precision',
                'Recall',
                ConfidenceScore()
            ],
            'confidence': ['mse', 'mae']
        }
    )
    
    # Load validation data
    logger.info("Loading validation data...")
    df = pd.read_csv('future_returns_dataset.csv')
    validation_size = len(df) // 5
    df_validation = df.iloc[-validation_size:].reset_index(drop=True)
    
    # Prepare features
    feature_manager = EnhancedFeatureManager(df_validation)
    X_dict, y_true = feature_manager.prepare_features(lookback=20)
    
    # Get predictions
    predictions, confidences = model.predict(
        [X_dict[group] for group in sorted(X_dict.keys())],
        batch_size=1024
    )
    
    # Convert predictions to binary
    y_pred_binary = (predictions > 0.5).astype(int)
    y_true = y_true.astype(int)
    
    # 1. Prediction Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(predictions, bins=50, alpha=0.7, label='Predictions')
    plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Boundary')
    plt.title('Distribution of Model Predictions')
    plt.xlabel('Prediction Value')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(f'{save_path}/prediction_distribution.png')
    plt.close()
    
    # 2. Confidence Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(confidences, bins=50, alpha=0.7, color='green')
    plt.title('Distribution of Model Confidence')
    plt.xlabel('Confidence Value')
    plt.ylabel('Count')
    plt.savefig(f'{save_path}/confidence_distribution.png')
    plt.close()
    
    # 3. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{save_path}/confusion_matrix.png')
    plt.close()
    
    # 4. ROC Curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_true, predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_path}/roc_curve.png')
    plt.close()
    
    # 5. Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_true, predictions)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(f'{save_path}/precision_recall_curve.png')
    plt.close()
    
    # 6. Confidence vs Accuracy
    confidence_bins = np.linspace(0, 1, 11)
    accuracy_by_confidence = []
    for i in range(len(confidence_bins)-1):
        mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
        if mask.sum() > 0:
            accuracy = accuracy_score(y_true[mask], y_pred_binary[mask])
            accuracy_by_confidence.append(accuracy)
    
    plt.figure(figsize=(12, 6))
    plt.plot(confidence_bins[1:], accuracy_by_confidence, marker='o')
    plt.title('Model Accuracy by Confidence Level')
    plt.xlabel('Confidence Level')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f'{save_path}/confidence_vs_accuracy.png')
    plt.close()
    
    # Save metrics to JSON
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary),
        'recall': recall_score(y_true, y_pred_binary),
        'f1': f1_score(y_true, y_pred_binary),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confidence_stats': {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }
    }
    
    with open(f'{save_path}/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Model visualization complete. Results saved in {save_path}")

def main():
    """Main function to run the enhanced training pipeline."""
    start_time = time.time()
    
    # Set environment variables for optimal performance
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '1'
    
    # Create results directory
    results_path = f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(results_path, exist_ok=True)
    
    # Load data in chunks
    logger.info("Loading data...")
    chunk_size = 10000
    chunks = []
    
    try:
        for chunk in pd.read_csv('future_returns_dataset.csv', chunksize=chunk_size):
            chunk, _ = parse_orderbook_data(chunk)
            chunks.append(chunk)
            logger.info(f"Loaded chunk of size {len(chunk)}")
            gc.collect()
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    logger.info("Concatenating chunks...")
    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Total rows loaded: {len(df)}")
    
    del chunks
    gc.collect()
    
    # Drop NaN values
    df = df.dropna()
    logger.info(f"Rows after dropping NaN: {len(df)}")
    
    # Initialize feature manager and prepare features
    logger.info("Initializing feature manager...")
    feature_manager = EnhancedFeatureManager(df)
    
    logger.info("Preparing features...")
    X_dict, y = feature_manager.prepare_features(lookback=20)
    
    del df
    gc.collect()
    
    # Train model with cross-validation
    all_histories, all_val_predictions, all_val_confidences, all_val_true = train_model(
        X_dict, 
        y, 
        n_splits=3,
        epochs=50,
        batch_size=1024
    )
    
    # Plot training history
    plot_training_history(all_histories, save_path=results_path)
    
    # Convert predictions to binary and plot confusion matrices
    binary_predictions = (np.array(all_val_predictions) > 0.5).astype(int)
    plot_confusion_matrix(
        np.array(all_val_true),
        binary_predictions,
        np.array(all_val_confidences),
        save_path=results_path
    )
    
    # Save best model
    best_val_loss = float('inf')
    best_fold = 0
    
    for fold, history in enumerate(all_histories):
        val_loss = min(history['val_prediction_loss'])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_fold = fold
    
    # Copy best model to final location
    best_model_path = f'model_fold_{best_fold}.h5'
    final_model_path = 'best_model_30min.keras'
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy2(best_model_path, final_model_path)
        logger.info(f"Best model (fold {best_fold}) saved as {final_model_path}")
    
    # Calculate execution time
    execution_time = time.time() - start_time
    logger.info(f"\nTotal execution time: {execution_time:.2f} seconds")
    
    # Save configuration and results
    config = {
        'feature_groups': feature_manager.feature_groups,
        'training_results': {
            'best_fold': best_fold,
            'best_val_loss': float(best_val_loss),
            'execution_time': execution_time,
            'parameters': {
                'lookback': 20,
                'batch_size': 1024,
                'epochs': 50,
                'n_splits': 3
            }
        }
    }
    
    with open(f'{results_path}/model_configuration.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Training complete. Results saved in {results_path}")

if __name__ == "__main__":
    main()
    visualize_saved_model_performance() 