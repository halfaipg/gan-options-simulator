"""
LSTM-based GAN model for DLV time series generation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from .base_model import BaseModel


class LSTMGAN(BaseModel):
    """
    LSTM-based Generative Adversarial Network for DLV time series generation.
    """
    
    def __init__(self, 
                 state_dim, 
                 noise_dim, 
                 output_dim, 
                 generator_units=[128, 256], 
                 discriminator_units=[256, 128],
                 use_pca=False,
                 n_pca_components=5,
                 log_dir='logs'):
        """
        Initialize LSTM-GAN model.
        
        Parameters:
        -----------
        state_dim : int or tuple
            Dimension of state input (batch_size, seq_length, features)
        noise_dim : int
            Dimension of noise input
        output_dim : int or tuple
            Dimension of output (DLVs)
        generator_units : list
            List of units in each LSTM layer of the generator
        discriminator_units : list
            List of units in each LSTM layer of the discriminator
        use_pca : bool
            Whether to use PCA compression for the output
        n_pca_components : int
            Number of PCA components if use_pca is True
        log_dir : str
            Directory for saving logs
        """
        self.state_dim = state_dim
        self.noise_dim = noise_dim
        self.output_dim = output_dim if not use_pca else n_pca_components
        self.generator_units = generator_units
        self.discriminator_units = discriminator_units
        self.use_pca = use_pca
        
        # Define input dimensions for the base model
        input_dim = (noise_dim + state_dim[1] * state_dim[2],)  # Combined noise and flattened state
        
        super().__init__(
            name=f'LSTM-GAN{"_PCA" if use_pca else ""}',
            input_dim=input_dim,
            output_dim=self.output_dim,
            log_dir=log_dir
        )
        
        # Build generator and discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Build complete model
        self.model = self.build_model()
        
        # Define optimizers
        self.generator_optimizer = None
        self.discriminator_optimizer = None
    
    def build_generator(self):
        """
        Build the generator network.
        
        Returns:
        --------
        Model
            Generator model
        """
        # Input for noise
        noise_input = layers.Input(shape=(self.noise_dim,), name='noise_input')
        
        # Input for state
        state_input = layers.Input(shape=self.state_dim[1:], name='state_input')
        
        # Flatten state input
        flattened_state = layers.Flatten()(state_input)
        
        # Concatenate noise and state
        combined_input = layers.Concatenate()([noise_input, flattened_state])
        
        # Dense layers for initial processing
        x = layers.Dense(self.generator_units[0], activation='relu')(combined_input)
        x = layers.BatchNormalization()(x)
        
        # Reshape for LSTM input
        # Assuming sequence length is 1 for the generated output
        x = layers.Reshape((1, self.generator_units[0]))(x)
        
        # LSTM layers
        for units in self.generator_units[1:]:
            x = layers.LSTM(units, return_sequences=True)(x)
            x = layers.BatchNormalization()(x)
        
        # Output layer
        if isinstance(self.output_dim, tuple):
            # Reshape to match the output dimension
            x = layers.Flatten()(x)
            x = layers.Dense(np.prod(self.output_dim))(x)
            x = layers.Reshape(self.output_dim)(x)
        else:
            # Direct output for PCA components or simple output
            x = layers.TimeDistributed(layers.Dense(self.output_dim))(x)
            x = layers.Flatten()(x)  # Flatten the time dimension
        
        # Create model
        generator = models.Model(inputs=[noise_input, state_input], outputs=x, name='generator')
        generator.summary()
        
        return generator
    
    def build_discriminator(self):
        """
        Build the discriminator network.
        
        Returns:
        --------
        Model
            Discriminator model
        """
        # Input for generated/real data
        if isinstance(self.output_dim, tuple):
            data_input = layers.Input(shape=self.output_dim, name='data_input')
            x = layers.Flatten()(data_input)
        else:
            data_input = layers.Input(shape=(self.output_dim,), name='data_input')
            x = data_input
        
        # Input for state
        state_input = layers.Input(shape=self.state_dim[1:], name='state_input')
        flattened_state = layers.Flatten()(state_input)
        
        # Concatenate data and state
        combined_input = layers.Concatenate()([x, flattened_state])
        
        # Dense processing
        x = layers.Dense(self.discriminator_units[0], activation='relu')(combined_input)
        x = layers.LayerNormalization()(x)
        
        # Reshape for LSTM
        x = layers.Reshape((1, self.discriminator_units[0]))(x)
        
        # LSTM layers
        for units in self.discriminator_units[1:]:
            x = layers.LSTM(units, return_sequences=True)(x)
            x = layers.LayerNormalization()(x)
        
        # Flatten and output
        x = layers.Flatten()(x)
        x = layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        discriminator = models.Model(inputs=[data_input, state_input], outputs=x, name='discriminator')
        discriminator.summary()
        
        return discriminator
    
    def build_model(self):
        """
        Build the complete GAN model.
        
        Returns:
        --------
        Model
            Complete GAN model
        """
        # For GANs, the complete model is primarily used for visualization and saving
        # The actual training is handled by the train_step method
        
        # Freeze the discriminator for the generator training
        self.discriminator.trainable = False
        
        # Connect the generator to the discriminator
        noise_input = layers.Input(shape=(self.noise_dim,), name='noise_input')
        state_input = layers.Input(shape=self.state_dim[1:], name='state_input')
        
        # Generate fake data
        generated_data = self.generator([noise_input, state_input])
        
        # Get discriminator output for the generated data
        validity = self.discriminator([generated_data, state_input])
        
        # Create and return the combined model
        model = models.Model(inputs=[noise_input, state_input], outputs=validity, name='lstmgan')
        return model
    
    def compile(self, generator_optimizer, discriminator_optimizer, loss_fn=None, metrics=None):
        """
        Compile the GAN model with optimizers and loss function.
        
        Parameters:
        -----------
        generator_optimizer : Optimizer
            Optimizer for the generator
        discriminator_optimizer : Optimizer
            Optimizer for the discriminator
        loss_fn : callable, optional
            Loss function (default is binary crossentropy)
        metrics : list of callable, optional
            Metrics to track during training
            
        Returns:
        --------
        self
            Returns self for method chaining
        """
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_fn = loss_fn or tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.metrics = metrics or []
        
        # Initialize metrics in history
        for metric in self.metrics:
            metric_name = metric.__name__ if hasattr(metric, '__name__') else str(metric)
            self.history['metrics'][metric_name] = []
        
        # Add GAN-specific metrics
        self.history['metrics']['gen_loss'] = []
        self.history['metrics']['disc_loss'] = []
        self.history['metrics']['disc_real_acc'] = []
        self.history['metrics']['disc_fake_acc'] = []
        
        return self
    
    def train_step(self, batch_data):
        """
        Perform a single training step.
        
        Parameters:
        -----------
        batch_data : tuple
            Batch of training data (state, next_state)
            
        Returns:
        --------
        dict
            Dictionary of loss values for the step
        """
        # Unpack batch data
        state, next_state = batch_data
        batch_size = tf.shape(state)[0]
        
        # Reshape next_state for discriminator input
        # Expected shape: (batch_size, output_dim)
        next_state_reshaped = tf.reshape(next_state, [-1, self.output_dim])

        # Train discriminator
        with tf.GradientTape() as disc_tape:
            # Generate fake data
            noise = tf.random.normal([batch_size, self.noise_dim])
            generated_data = self.generator([noise, state], training=False)
            
            # Get discriminator outputs
            real_output = self.discriminator([next_state_reshaped, state], training=True)
            fake_output = self.discriminator([generated_data, state], training=True)
            
            # Calculate discriminator loss
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
        
        # Apply discriminator gradients
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        
        # Train the generator
        with tf.GradientTape() as gen_tape:
            # Generate fake data
            generated_data = self.generator([noise, state], training=True)
            
            # Get discriminator output for the generated data
            fake_output = self.discriminator([generated_data, state], training=True)
            
            # Calculate generator loss
            # The generator wants the discriminator to classify fake data as real
            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
        
        # Apply generator gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        
        # Calculate accuracy metrics
        disc_real_acc = tf.reduce_mean(tf.cast(real_output > 0.5, tf.float32))
        disc_fake_acc = tf.reduce_mean(tf.cast(fake_output < 0.5, tf.float32))
        
        # Return metrics
        return {
            'loss': disc_loss + gen_loss,
            'gen_loss': gen_loss,
            'disc_loss': disc_loss,
            'disc_real_acc': disc_real_acc,
            'disc_fake_acc': disc_fake_acc
        }
    
    def validate_step(self, batch_data):
        """
        Perform a single validation step.
        
        Parameters:
        -----------
        batch_data : tuple
            Batch of validation data (state, next_state)
            
        Returns:
        --------
        dict
            Dictionary of loss values for the validation step
        """
        state, next_state = batch_data
        batch_size = tf.shape(state)[0]

        # Reshape next_state for discriminator input
        next_state_reshaped = tf.reshape(next_state, [-1, self.output_dim])

        # Generate fake data
        noise = tf.random.normal([batch_size, self.noise_dim])
        generated_data = self.generator([noise, state], training=False)

        # Get discriminator outputs
        real_output = self.discriminator([next_state_reshaped, state], training=False)
        fake_output = self.discriminator([generated_data, state], training=False)

        # Calculate losses
        disc_real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        disc_fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        disc_loss = (disc_real_loss + disc_fake_loss) / 2
        gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)

        # Calculate accuracies (optional, but good for monitoring)
        real_acc = tf.reduce_mean(tf.cast(real_output >= 0.5, tf.float32))
        fake_acc = tf.reduce_mean(tf.cast(fake_output < 0.5, tf.float32))

        return {
            'loss': disc_loss + gen_loss,  # Combined loss for basic tracking
            'gen_loss': gen_loss,
            'disc_loss': disc_loss,
            'disc_real_acc': real_acc,
            'disc_fake_acc': fake_acc
        }
    
    def generate(self, state, n_samples=1):
        """
        Generate samples from the model.
        
        Parameters:
        -----------
        state : ndarray
            Current state for conditional generation with shape matching self.state_dim
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        ndarray
            Generated samples
        """
        # Create random noise
        noise = tf.random.normal([n_samples, self.noise_dim])
        
        # Ensure state has the right shape
        if len(state.shape) == 2:  # (seq_length, features)
            state = np.expand_dims(state, axis=0)  # Add batch dimension
        
        # Repeat state for multiple samples if needed
        if n_samples > 1 and state.shape[0] == 1:
            state = np.repeat(state, n_samples, axis=0)
        
        # Generate samples
        generated_data = self.generator.predict([noise, state])
        
        return generated_data
    
    def generate_sequences(self, initial_state, sequence_length, use_generated_state=True):
        """
        Generate a sequence of DLV data.
        
        Parameters:
        -----------
        initial_state : ndarray
            Initial state for the sequence generation
        sequence_length : int
            Length of the sequence to generate
        use_generated_state : bool
            If True, use the generated data as part of the next state;
            if False, only use the initial state shifted by one step
            
        Returns:
        --------
        ndarray
            Generated sequence with shape (sequence_length, output_dim)
        """
        # Ensure initial state has the right shape
        if len(initial_state.shape) == 2:  # (seq_length, features)
            initial_state = np.expand_dims(initial_state, axis=0)  # Add batch dimension
        
        # Initialize sequence with the first state
        generated_sequence = []
        current_state = initial_state.copy()
        
        for _ in range(sequence_length):
            # Generate the next step
            next_step = self.generate(current_state, n_samples=1)[0]  # Get the first sample
            generated_sequence.append(next_step)
            
            if use_generated_state:
                # Update state with the generated data
                if isinstance(self.output_dim, tuple):
                    # Reshape generated data to match state feature dimension
                    reshaped_data = next_step.reshape(1, 1, -1)  # (1, 1, features)
                else:
                    # Expand dimensions for the sequence
                    reshaped_data = np.expand_dims(next_step, axis=(0, 1))  # (1, 1, n_components)
                
                # Shift state and add the new data
                current_state = np.roll(current_state, shift=-1, axis=1)
                current_state[:, -1, :] = reshaped_data[:, 0, :]
            else:
                # Just shift the initial state without using generated data
                current_state = np.roll(initial_state, shift=-1, axis=1)
        
        return np.array(generated_sequence) 