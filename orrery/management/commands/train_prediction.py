import json
import os
import numpy as np
from django.core.management.base import BaseCommand
from django.conf import settings
from orrery.models import Planet, Comet, Asteroid
from sklearn.linear_model import LogisticRegression

class Command(BaseCommand):
    help = 'Train the probability prediction model using database data'

    def handle(self, *args, **options):
        self.stdout.write("Collecting data from database...")
        
        data = []
        labels = []
        
        # Mapping for labels
        # 0: Asteroid, 1: Comet, 2: Planet, 3: PHA
        label_names = ['Asteroid', 'Comet', 'Planet', 'PHA']
        
        # Load Planets
        planets = Planet.objects.exclude(size__isnull=True).exclude(distance__isnull=True)
        for p in planets:
            # Planets: size is km, distance is km
            data.append([p.size, p.distance])
            labels.append(2)
        
        # Load Comets
        comets = Comet.objects.exclude(distance__isnull=True)
        for c in comets:
            # Heuristic: If size is missing, assume average comet size ~10km
            size = c.size if c.size is not None else 10.0
            data.append([size, c.distance])
            labels.append(1)
                
        # Load Asteroids
        asteroids = Asteroid.objects.exclude(size__isnull=True).exclude(distance__isnull=True)
        for a in asteroids:
            # Asteroids: size is meters, distance is km
            data.append([a.size / 1000.0, a.distance])
            if a.is_potentially_hazardous:
                labels.append(3)
            else:
                labels.append(0)
        
        if not data:
            self.stdout.write(self.style.ERROR("No data found for training."))
            return

        X = np.array(data)
        y = np.array(labels)
        
        unique_labels = np.unique(y)
        self.stdout.write(f"Training on {len(data)} samples with labels {unique_labels}...")
        
        # Scaling factors used in JS: size / 1000, distance / 1000000
        X_scaled = X.copy()
        X_scaled[:, 0] = X_scaled[:, 0] / 1000.0
        X_scaled[:, 1] = X_scaled[:, 1] / 1000000.0
        
        # Train Logistic Regression with Balanced Class Weights
        # This prevents the 35k asteroids from overwhelming the 8 planets.
        model = LogisticRegression(
            multi_class='multinomial', 
            solver='lbfgs', 
            max_iter=2000,
            class_weight='balanced'
        )
        model.fit(X_scaled, y)
        
        weights = model.coef_.tolist()
        biases = model.intercept_.tolist()
        
        trained_labels = model.classes_.tolist()
        label_map = {label: i for i, label in enumerate(trained_labels)}
        
        # JS Order: Asteroid (0), Comet (1), Meteor (2), Planet (3), PHA (4)
        # Neutral defaults - close to zero so they don't dominate
        final_weights = [
            [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
        ]
        final_biases = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Map our trained classes: 0:Asteroid, 1:Comet, 2:Planet, 3:PHA
        mapping = {
            0: 0, # Asteroid -> index 0
            1: 1, # Comet -> index 1
            2: 3, # Planet -> index 3
            3: 4  # PHA -> index 4
        }
        
        for label_val, js_idx in mapping.items():
            if label_val in label_map:
                model_idx = label_map[label_val]
                final_weights[js_idx] = weights[model_idx]
                final_biases[js_idx] = biases[model_idx]
            else:
                # Meteor (2) is not in DB, give it some manual "scary" fallback
                if js_idx == 2:
                    final_weights[js_idx] = [-0.1, -0.05]
                    final_biases[js_idx] = -2.0
        
        # Calculate accuracy on training set
        accuracy = model.score(X_scaled, y)
        self.stdout.write(f"Model accuracy: {accuracy:.4f}")
        
        model_data = {
            'weights': final_weights,
            'biases': final_biases,
            'classes': ['Asteroid', 'Comet', 'Meteor', 'Planet', 'PHA'],
            'accuracy': accuracy
        }
        
        # Save to file in a location accessible by views
        output_dir = os.path.join(settings.BASE_DIR, 'orrery', 'data')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'model_params.json')
        
        with open(output_path, 'w') as f:
            json.dump(model_data, f)
            
        self.stdout.write(self.style.SUCCESS(f"Successfully trained model with {accuracy:.2%} accuracy and saved parameters to {output_path}"))
