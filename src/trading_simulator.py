import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
import gc
import json
import scipy.stats
from scipy import stats
from collections import deque, defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"trading_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set mixed precision policy
tf.keras.mixed_precision.set_global_policy('float32')

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
        
        # Calculate metrics
        true_positives = tf.reduce_sum(y_true * y_pred)
        false_positives = tf.reduce_sum((1 - y_true) * y_pred)
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred))
        
        # Update state
        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

@tf.keras.saving.register_keras_serializable()
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

class EnhancedTradingSimulator:
    def __init__(self, initial_capital=100000.0, base_position_size=0.1, stop_loss=0.02):
        self.initial_capital = initial_capital
        self.base_position_size = base_position_size
        self.stop_loss = stop_loss
        self.setup_logging()
        
        # Advanced risk parameters
        self.max_position_size = 0.2
        self.min_confidence_threshold = 0.4
        self.volatility_target = 0.15
        self.max_leverage = 2.0
        self.position_timeout = 20  # bars
        
        # Performance tracking
        self.position_history = []
        self.trade_stats = defaultdict(list)
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(self, confidence, volatility, current_exposure):
        """Calculate position size with advanced risk management."""
        # Base sizing
        vol_scalar = self.volatility_target / volatility if volatility > 0 else 1
        conf_scalar = self._calculate_confidence_scalar(confidence)
        
        # Dynamic position sizing
        raw_size = self.base_position_size * vol_scalar * conf_scalar
        
        # Apply portfolio constraints
        position_size = min(
            raw_size,
            self.max_position_size,
            self.max_leverage - abs(current_exposure)
        )
        
        return position_size
    
    def _calculate_confidence_scalar(self, confidence):
        """Calculate confidence-based position scaling."""
        if confidence >= 0.8:
            return 1.0
        elif confidence >= 0.6:
            return 0.7
        elif confidence >= self.min_confidence_threshold:
            return 0.4
        return 0.0
    
    def simulate_trading(self, predictions, confidences, returns, dates):
        """Enhanced trading simulation with advanced risk management."""
        self.logger.info("Starting advanced trading simulation...")
        
        # Initialize tracking
        capital = self.initial_capital
        positions = pd.Series(index=dates, dtype=float)
        daily_pnl = pd.Series(index=dates, dtype=float)
        cumulative_returns = pd.Series(index=dates, dtype=float)
        
        # Risk management
        volatility_window = 20
        rolling_volatility = pd.Series(returns).rolling(window=volatility_window).std()
        rolling_correlation = pd.Series(returns).rolling(window=volatility_window).corr(
            pd.Series(predictions)
        )
        
        # Performance tracking
        trade_count = 0
        winning_trades = 0
        max_drawdown = 0
        peak_capital = capital
        current_position_time = 0
        
        for i in range(len(dates)):
            current_date = dates[i]
            
            if i < volatility_window:
                continue
            
            # Advanced position sizing
            current_volatility = rolling_volatility.iloc[i]
            current_correlation = rolling_correlation.iloc[i]
            current_exposure = positions.iloc[i-1] / capital if i > 0 else 0
            
            position_size = self.calculate_position_size(
                confidences[i],
                current_volatility,
                current_exposure
            )
            
            # Generate trading signal with dynamic thresholds
            confidence_threshold = 0.5 + (1 - confidences[i]) * 0.1
            if predictions[i] > confidence_threshold:
                signal = 1
            elif predictions[i] < (1 - confidence_threshold):
                signal = -1
            else:
                signal = 0
            
            # Apply correlation-based signal adjustment
            if abs(current_correlation) > 0.3:
                signal *= np.sign(current_correlation)
            
            # Position sizing with capital allocation
            position_value = capital * position_size * signal
            
            # Apply position timeout
            if current_position_time >= self.position_timeout:
                position_value = 0
                current_position_time = 0
            
            positions.iloc[i] = position_value
            
            # Calculate returns and update metrics
            if i > 0:
                trade_return = returns[i] * positions.iloc[i-1] / capital
                daily_pnl.iloc[i] = trade_return * capital
                
                # Update capital and track metrics
                capital += daily_pnl.iloc[i]
                cumulative_returns.iloc[i] = (capital / self.initial_capital) - 1
                
                # Track drawdown
                peak_capital = max(peak_capital, capital)
                current_drawdown = (peak_capital - capital) / peak_capital
                max_drawdown = max(max_drawdown, current_drawdown)
                
                # Track trades
                if positions.iloc[i] != positions.iloc[i-1]:
                    trade_count += 1
                    if daily_pnl.iloc[i] > 0:
                        winning_trades += 1
                    
                    # Store trade statistics
                    self.trade_stats['confidence'].append(confidences[i])
                    self.trade_stats['return'].append(trade_return)
                    self.trade_stats['position_size'].append(position_size)
                    self.trade_stats['volatility'].append(current_volatility)
                
                # Update position time
                if positions.iloc[i] != 0:
                    current_position_time += 1
                else:
                    current_position_time = 0
                
                # Log significant moves
                if abs(daily_pnl.iloc[i] / capital) > 0.03:
                    self._log_significant_move(
                        current_date,
                        trade_return,
                        daily_pnl.iloc[i],
                        signal,
                        predictions[i],
                        confidences[i],
                        position_size,
                        current_volatility,
                        current_correlation
                    )
            
            # Risk management
            if abs(cumulative_returns.iloc[i]) <= -self.stop_loss:
                positions.iloc[i] = 0
                current_position_time = 0
                self.logger.warning(
                    f"Stop loss triggered at {current_date}"
                    f"\nConfidence: {confidences[i]:.4f}"
                    f"\nDrawdown: {current_drawdown:.4f}"
                )
            
            # Memory management
            if i % 1000 == 0:
                gc.collect()
        
        # Calculate final metrics
        performance_metrics = self._calculate_performance_metrics(
            capital,
            daily_pnl,
            trade_count,
            winning_trades,
            max_drawdown
        )
        
        # Calculate advanced analytics
        analytics = self._calculate_advanced_analytics(
            positions,
            returns,
            predictions,
            confidences
        )
        
        return {
            'capital': capital,
            'positions': positions,
            'daily_pnl': daily_pnl,
            'cumulative_returns': cumulative_returns,
            'metrics': performance_metrics,
            'analytics': analytics,
            'trade_stats': self.trade_stats
        }
    
    def _log_significant_move(self, date, trade_return, pnl, signal, prediction,
                            confidence, position_size, volatility, correlation):
        """Log detailed information about significant market moves."""
        self.logger.info(
            f"\nSignificant move detected at {date}:"
            f"\nReturn: {trade_return:.4f}"
            f"\nPnL: ${pnl:.2f}"
            f"\nPosition: {'LONG' if signal > 0 else 'SHORT' if signal < 0 else 'NEUTRAL'}"
            f"\nPrediction: {prediction:.4f}"
            f"\nConfidence: {confidence:.4f}"
            f"\nPosition Size: {position_size:.4f}"
            f"\nVolatility: {volatility:.4f}"
            f"\nPrediction-Return Correlation: {correlation:.4f}"
        )
    
    def _calculate_performance_metrics(self, capital, daily_pnl, trade_count,
                                    winning_trades, max_drawdown):
        """Calculate comprehensive performance metrics."""
        win_rate = winning_trades / trade_count if trade_count > 0 else 0
        avg_daily_return = daily_pnl.mean()
        daily_sharpe = daily_pnl.mean() / daily_pnl.std() if daily_pnl.std() > 0 else 0
        
        # Calculate advanced metrics
        daily_returns = daily_pnl / self.initial_capital
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        calmar_ratio = self._calculate_calmar_ratio(daily_returns, max_drawdown)
        
        return {
            'win_rate': win_rate,
            'trade_count': trade_count,
            'avg_daily_return': avg_daily_return,
            'daily_sharpe': daily_sharpe,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'final_return': (capital/self.initial_capital - 1),
            'returns_stats': {
                'skew': stats.skew(daily_pnl.dropna()),
                'kurtosis': stats.kurtosis(daily_pnl.dropna())
            }
        }
    
    def _calculate_advanced_analytics(self, positions, returns, predictions, confidences):
        """Calculate advanced trading analytics."""
        # Position analysis
        position_changes = positions.diff().fillna(0)
        turnover = abs(position_changes).sum() / self.initial_capital
        
        # Prediction analysis
        prediction_accuracy = np.mean(
            (predictions > 0.5) == (returns > 0)
        )
        
        # Confidence analysis
        high_conf_mask = confidences >= 0.8
        med_conf_mask = (confidences >= 0.6) & (confidences < 0.8)
        low_conf_mask = (confidences >= 0.4) & (confidences < 0.6)
        
        confidence_metrics = {
            'high_confidence': {
                'count': int(high_conf_mask.sum()),
                'accuracy': float(prediction_accuracy[high_conf_mask].mean())
            },
            'medium_confidence': {
                'count': int(med_conf_mask.sum()),
                'accuracy': float(prediction_accuracy[med_conf_mask].mean())
            },
            'low_confidence': {
                'count': int(low_conf_mask.sum()),
                'accuracy': float(prediction_accuracy[low_conf_mask].mean())
            }
        }
        
        return {
            'turnover': turnover,
            'prediction_accuracy': prediction_accuracy,
            'confidence_metrics': confidence_metrics,
            'position_concentration': abs(positions).mean() / self.initial_capital,
            'avg_holding_period': len(positions) / max(1, len(position_changes[position_changes != 0]))
        }
    
    def _calculate_sortino_ratio(self, returns, risk_free_rate=0.02/252):
        """Calculate Sortino ratio using downside deviation."""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
        return np.sqrt(252) * np.mean(excess_returns) / downside_std
    
    def _calculate_calmar_ratio(self, returns, max_drawdown, periods=252):
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        annualized_return = np.mean(returns) * periods
        return abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0

def load_and_prepare_model(model_path):
    """Load the trained model with custom objects and mixed precision handling."""
    try:
        # Disable mixed precision temporarily during model loading
        original_policy = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy('float32')
        
        # Define custom objects with proper type handling
        custom_objects = {
            'F1Score': F1Score,
            'ConfidenceScore': ConfidenceScore
        }
        
        # Load model with custom objects
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        
        # Restore original mixed precision policy
        tf.keras.mixed_precision.set_global_policy(original_policy)
        
        # Compile model with appropriate settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy', F1Score(), ConfidenceScore()]
        )
        
        return model
            
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def plot_trading_results(simulator, results, save_path='trading_results'):
    """Plot comprehensive trading results with confidence analysis."""
    os.makedirs(save_path, exist_ok=True)
    
    # Create subplots for different metrics
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2)
    
    # 1. Portfolio Value Plot
    ax1 = fig.add_subplot(gs[0, :])
    initial_capital = simulator.initial_capital
    portfolio_values = initial_capital * (1 + results['cumulative_returns'])
    ax1.plot(portfolio_values.index, portfolio_values.values, label='Portfolio Value', color='blue')
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Daily Returns Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    daily_returns = results['daily_pnl'] / simulator.initial_capital
    sns.histplot(daily_returns.dropna(), kde=True, ax=ax2)
    ax2.set_title('Distribution of Daily Returns')
    ax2.set_xlabel('Daily Return')
    ax2.set_ylabel('Frequency')
    
    # 3. Confidence Analysis
    ax3 = fig.add_subplot(gs[1, 1])
    conf_metrics = results['metrics']['confidence_metrics']
    conf_data = [
        conf_metrics['high_confidence']['count'],
        conf_metrics['medium_confidence']['count'],
        conf_metrics['low_confidence']['count']
    ]
    ax3.bar(['High', 'Medium', 'Low'], conf_data)
    ax3.set_title('Trade Distribution by Confidence Level')
    ax3.set_xlabel('Confidence Level')
    ax3.set_ylabel('Number of Trades')
    
    # 4. Rolling Metrics
    ax4 = fig.add_subplot(gs[2, 0])
    rolling_sharpe = (
        daily_returns.rolling(window=20).mean() / 
        daily_returns.rolling(window=20).std() * 
        np.sqrt(252)
    )
    ax4.plot(rolling_sharpe.index, rolling_sharpe.values, label='Rolling Sharpe')
    ax4.set_title('20-Day Rolling Sharpe Ratio')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.grid(True)
    ax4.legend()
    
    # 5. Drawdown Analysis
    ax5 = fig.add_subplot(gs[2, 1])
    rolling_max = portfolio_values.expanding().max()
    drawdowns = (portfolio_values - rolling_max) / rolling_max
    ax5.fill_between(drawdowns.index, drawdowns.values, 0, color='red', alpha=0.3)
    ax5.set_title('Portfolio Drawdown')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Drawdown %')
    ax5.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/trading_analysis.png')
    plt.close()
    
    # Create additional plots
    
    # Position Analysis
    plt.figure(figsize=(15, 7))
    positions = results['positions']
    plt.plot(positions.index, positions.values, label='Position Size', color='green')
    plt.title('Position Sizes Over Time')
    plt.xlabel('Date')
    plt.ylabel('Position Value ($)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{save_path}/position_analysis.png')
    plt.close()
    
    # Performance Metrics Table
    plt.figure(figsize=(12, 8))
    metrics = results['metrics']
    metric_data = {
        'Metric': [
            'Total Return (%)',
            'Sharpe Ratio',
            'Max Drawdown (%)',
            'Win Rate (%)',
            'Number of Trades',
            'Avg Daily Return ($)'
        ],
        'Value': [
            f"{((portfolio_values.iloc[-1]/simulator.initial_capital - 1) * 100):.2f}",
            f"{metrics['daily_sharpe']:.2f}",
            f"{metrics['max_drawdown']*100:.2f}",
            f"{metrics['win_rate']*100:.2f}",
            f"{metrics['trade_count']}",
            f"{metrics['avg_daily_return']:.2f}"
        ]
    }
    
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(
        cellText=list(zip(metric_data['Metric'], metric_data['Value'])),
        colLabels=['Metric', 'Value'],
        cellLoc='center',
        loc='center',
        colWidths=[0.6, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    plt.title('Performance Metrics Summary', pad=20)
    plt.savefig(f'{save_path}/performance_metrics.png')
    plt.close()
    
    # Save detailed results to JSON
    detailed_results = {
        'portfolio_metrics': {
            'final_value': float(portfolio_values.iloc[-1]),
            'total_return': float(portfolio_values.iloc[-1]/simulator.initial_capital - 1),
            'max_drawdown': float(metrics['max_drawdown']),
            'sharpe_ratio': float(metrics['daily_sharpe']),
            'win_rate': float(metrics['win_rate'])
        },
        'confidence_analysis': {
            'high_confidence_trades': int(conf_metrics['high_confidence']['count']),
            'medium_confidence_trades': int(conf_metrics['medium_confidence']['count']),
            'low_confidence_trades': int(conf_metrics['low_confidence']['count'])
        },
        'trade_metrics': {
            'total_trades': int(metrics['trade_count']),
            'average_daily_return': float(metrics['avg_daily_return']),
            'return_skewness': float(metrics['returns_stats']['skew']),
            'return_kurtosis': float(metrics['returns_stats']['kurtosis'])
        }
    }
    
    with open(f'{save_path}/detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=4)

def main():
    """Main function to run the trading simulation."""
    # Load the trained model
    model = load_and_prepare_model('best_model_30min.keras')
    
    # Initialize simulator
    simulator = EnhancedTradingSimulator(
        initial_capital=100000.0,
        base_position_size=0.1,
        stop_loss=0.02
    )
    
    # Load and preprocess validation data
    logger.info("Loading and preprocessing validation data...")
    try:
        # Load data in chunks
        chunk_size = 10000
        chunks = []
        for chunk in pd.read_csv('future_returns_dataset.csv', chunksize=chunk_size):
            chunks.append(chunk)
            gc.collect()
        
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Total rows loaded: {len(df)}")
        
        # Get validation set (last 20% of data)
        validation_size = len(df) // 5
        df_validation = df.iloc[-validation_size:].reset_index(drop=True)
        
        # Prepare features using the same logic as in enhanced_classifier.py
        feature_manager = EnhancedFeatureManager(df_validation)
        X_dict, returns = feature_manager.prepare_features(lookback=20)
        
        # Get predictions and confidences from the enhanced model
        predictions, confidences = model.predict(
            [X_dict[group] for group in sorted(X_dict.keys())],
            batch_size=1000
        )
        
        # Ensure predictions and confidences are properly shaped
        predictions = predictions.flatten()
        confidences = confidences.flatten()
        
        # Get dates for the validation period
        dates = df_validation.index[-len(predictions):]
        
        # Run simulation with both predictions and confidences
        results = simulator.simulate_trading(
            predictions,
            confidences,
            returns[-len(predictions):],
            dates
        )
        
        # Plot and save results
        plot_trading_results(simulator, results)
        
        logger.info("Trading simulation completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in trading simulation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 