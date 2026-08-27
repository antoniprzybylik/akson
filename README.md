<div align="center">
  <img src="./assets/logo.png" alt="Logo of the Akson software package" width="50%">
</div>

# Akson

Akson is a Python software package for the efficient design of control systems. It supports the simulation of closed-loop systems comprising a plant and a controller, and also provides tools for exporting designed PID controllers directly as IEC 61131-3 Structured Text for deployment on programmable logic controllers.

## 📂 Usage examples

The following examples can be found in the [`examples/`](./examples/) directory:

- **SISO Plant Step Response** – A brief introduction to the package, computing the step response of a system built from LTI state-space matrices.
- **DC Engine Simulation** – Simulates the Maxon RE 40 motor, reproducing its catalogue steady-state speed and response dynamics.
- **SISO Plant and PID Controller** – Constructs and simulates a closed-loop system comprising a SISO plant and a PID controller.
- **SISO Plant and DMC Controller** – Constructs and simulates a closed-loop system comprising a SISO plant and a DMC controller.
- **Wood-Berry Column Control** - Demonstrates control of Wood-Berry column with many types of controllers (DMC, QDMC, PID). The Wood-Berry column is a canonical benchmark for linear MIMO systems.
- **Exporting PID Controller PLC Code** - Demonstrates eexporting of PLC code for PID controller.

## 📦 Building the package

This project uses [Poetry](https://python-poetry.org/) for packaging and dependency management. To build the source distribution and wheel run:

```bash
poetry build
```

The resulting packages will be placed in `dist/`.

## 📋 Running the tests

After installing the package, run tests with:

```
pytest
```

## 📄 License

This work is released into the public domain. You can use it without restriction.

## 👤 Author

**Antoni Michał Przybylik**  
📧 [antoni@taon.io](mailto:antoni@taon.io)  
🔗 [https://github.com/antoniprzybylik](https://github.com/antoniprzybylik)
