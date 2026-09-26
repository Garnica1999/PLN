# V13 · P1c — ¿Señal lingüística o tabla de identidades?

Los tokens mas asociados a cada clase en P1a son **nombres de cuenta** (`matthunter` en 13-17, `dannapaola` en 18-24, `wilvercuellar` en 35-49, `tiagemita` en 65-xx). Este experimento separa ese componente del resto sin reentrenar el meta-modelo.

## 1. Método

Se reconstruye el bloque de P1a tres veces cambiando solo el vocabulario permitido: el completo, el que excluye los tokens que aparecen en el corpus como mención (`@nombre`) y el que excluye además los hashtags. Los identificadores se detectan con las expresiones regulares del propio proyecto y sobre el train de los folds exteriores, no sobre el corpus entero.

## 2. Control block-only

| Variante | LogReg | LinearSVC |
|---|---|---|
| completo | 0.3997 | 0.3917 |
| sin_cuentas | 0.3410 | 0.3319 |
| sin_cuentas_ni_hashtags | 0.2992 | 0.2914 |

Suelo de azar: 0.1667.

## 3. Lectura

- El bloque completo da 0.3997 y sin cuentas 0.3410: la senal cae **0.0587**, es decir un **25.2 %** de la mejora sobre el azar.
- Identificadores detectados en el train exterior: 33,519 nombres de cuenta mencionados y 17,405 hashtags.
- **Veredicto: el bloque es sobre todo evidencia léxica**: excluir las cuentas deja intacta la mayor parte de la señal, que es lo que la hipótesis original esperaba.

## 4. Por qué importa aunque la señal sea real

Reconocer una cuenta **es** información legítima sobre la edad del autor (dice en qué comunidad escribe), y el protocolo lo respeta: todo se calcula dentro del fold y el test no se toca. Pero es una señal frágil y de naturaleza distinta: no describe cómo escribe la persona sino a quién menciona, y no se transfiere a cuentas no vistas. Por eso se mide por separado en vez de esconderlo dentro de una única cifra de macro-F1.

