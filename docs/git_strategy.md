# Estrategia de Git

Este proyecto usa **GitHub Flow**, simplificado a dos ramas de larga vida:

- `main`: código estable, listo para release. Solo recibe merges vía Pull Request desde `development`.
- `development`: rama de integración. Todas las ramas de feature nacen de aquí y regresan aquí vía PR.

## Flujo de trabajo

1. Crear una rama de feature desde `development`:
   `git checkout -b feature/nombre-corto development`
2. Hacer commits pequeños y descriptivos (convención sugerida: `feat:`, `fix:`, `docs:`, `refactor:`).
3. Abrir un Pull Request de `feature/nombre-corto` hacia `development`, describiendo qué cambia y por qué.
4. Revisar y mergear con **merge commit** (no squash), para conservar el historial de cambios.
5. Cuando `development` está estable y completa para una entrega, abrir un PR de `development` hacia `main`.
6. Al mergear a `main`, crear un **tag/release** siguiendo versionado semántico (`v1.0.0`).

## Versionado

- `v1.0.0`: primera entrega del proyecto (Curso II).
- Si se reutiliza este repo para otro curso de la especialización, la siguiente entrega se etiqueta `v2.0.0`, y así sucesivamente.

## Buenas prácticas aplicadas

- Uso de `.gitkeep` en carpetas provisionales (`data/`, `models/`, `mlruns/`) que se ignoran en `.gitignore`.
- Al menos un Pull Request cerrado exitosamente antes de la entrega final.
- Sin commits adicionales después de la fecha de entrega (30 de agosto de 2026).
