Here is a detailed documentation for the configuration JSON schema in Markdown format using expandable details tabs for each section:

```markdown
# Simulation Configuration Schema

This document provides a detailed explanation of each section in the simulation configuration JSON schema.

---

## **1. Simulation Name**
<details>
<summary><strong>Description</strong></summary>

Defines a descriptive name for the simulation configuration.

**Schema**:
```json
{
  "type": "string",
  "description": "A descriptive name for the simulation configuration."
}
```

**Example**:
```json
"name": "Test Simulation Configuration"
```

</details>

---

## **2. Inputs**
<details>
<summary><strong>Description</strong></summary>

Specifies the input meshes and their associated configurations.

**Schema**:
```json
{
  "type": "array",
  "minItems": 1,
  "description": "List of input meshes and their associated configurations.",
  "items": {
    "type": "object",
    "properties": {
      "path": { "type": "string", "description": "File path to the mesh used in the simulation." },
      "percent_fixed": { "type": "number", "minimum": 0, "maximum": 100, "description": "Percentage of fixed nodes." },
      "material": { "type": "string", "description": "Material name, referencing the 'materials' section." },
      "transform": { ... },
      "force": { ... }
    }
  }
}
```

### Subsections
#### a. Path
```json
"path": "meshes/rectangle.mesh"
```

#### b. Percent Fixed
Percentage of nodes fixed in the mesh:
```json
"percent_fixed": 50
```

#### c. Material
References a material defined in the `materials` section:
```json
"material": "steel"
```

#### d. Transform
Describes scaling, rotation, and translation:
```json
"transform": {
  "scale": [1.0, 1.0, 1.0],
  "rotation": [0.0, 0.0, 0.0, 1.0],
  "translation": [0.0, 0.0, 0.0]
}
```

#### e. Force
Defines forces acting on the mesh:
```json
"force": {
  "gravity": 9.81,
  "top_force": 100,
  "side_force": 50
}
```

</details>

---

## **3. Materials**
<details>
<summary><strong>Description</strong></summary>

Defines materials with physical properties.

**Schema**:
```json
{
  "type": "array",
  "minItems": 1,
  "items": {
    "type": "object",
    "properties": {
      "name": { "type": "string", "description": "Unique identifier for the material." },
      "density": { "type": "number", "minimum": 0, "description": "Density in kg/m³." },
      "young_modulus": { ... },
      "poisson_ratio": { ... },
      "color": { ... }
    }
  }
}
```

### Example
```json
"materials": [
  {
    "name": "steel",
    "density": 7850,
    "young_modulus": 2e11,
    "poisson_ratio": 0.3,
    "color": [255, 0, 0, 1]
  }
]
```

</details>

---

## **4. Friction**
<details>
<summary><strong>Description</strong></summary>

Defines friction-related parameters.

**Schema**:
```json
{
  "type": "object",
  "properties": {
    "friction_coefficient": { "type": "number", "minimum": 0, "maximum": 1 },
    "damping_coefficient": { "type": "number", "minimum": 0 }
  }
}
```

### Example
```json
"friction": {
  "friction_coefficient": 0.5,
  "damping_coefficient": 0.01
}
```

</details>

---

## **5. Simulation Parameters**
<details>
<summary><strong>Description</strong></summary>

Contains general simulation parameters.

**Schema**:
```json
{
  "type": "object",
  "properties": {
    "dhat": { "type": "number", "minimum": 0, "description": "Collision detection threshold." },
    "dmin": { "type": "number", "minimum": 0, "description": "Minimum distance to avoid penetration." },
    "dt": { "type": "number", "minimum": 0, "description": "Time step in seconds." }
  }
}
```

### Example
```json
"simulation": {
  "dhat": 0.001,
  "dmin": 0.0001,
  "dt": 0.0167
}
```

</details>

---

## **6. Communication**
<details>
<summary><strong>Description</strong></summary>

Specifies inter-process communication settings.

**Schema**:
```json
{
  "type": "object",
  "properties": {
    "method": { "type": "string", "enum": ["redis", "websocket", "grpc"] },
    "settings": { ... }
  }
}
```

### Example
```json
"communication": {
  "method": "redis",
  "settings": {
    "redis": {
      "host": "localhost",
      "port": 6379,
      "db": 0
    }
  }
}
```

</details>

---

## **7. Serialization**
<details>
<summary><strong>Description</strong></summary>

Defines data serialization settings.

**Schema**:
```json
{
  "type": "object",
  "properties": {
    "default_method": { "type": "string", "enum": ["json", "pickle", "bson"] }
  }
}
```

### Example
```json
"serialization": {
  "default_method": "json"
}
```

</details>

---

## **8. Optimizer**
<details>
<summary><strong>Description</strong></summary>

Specifies optimizer settings.

**Schema**:
```json
{
  "type": "object",
  "properties": {
    "type": { "type": "string", "enum": ["newton", "gradient_descent"] },
    "params": { ... }
  }
}
```

### Example
```json
"optimizer": {
  "type": "newton",
  "params": {
    "max_iterations": 100,
    "rtol": 1e-5,
    "n_threads": 4
  }
}
```

</details>

---

## **9. Logging**
<details>
<summary><strong>Description</strong></summary>

Defines logging configuration.

**Schema**:
```json
{
  "type": "object",
  "properties": {
    "level": { "type": "string", "enum": ["DEBUG", "INFO", "WARNING"] },
    "format": { "type": "string" },
    "handlers": { ... }
  }
}
```

### Example
```json
"logging": {
  "level": "INFO",
  "format": "%(asctime)s [%(levelname)s]: %(message)s",
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "level": "DEBUG"
    }
  }
}
```

</details>
```

This format includes a detailed explanation of each section, expandable for better readability. Let me know if you'd like further refinements!
