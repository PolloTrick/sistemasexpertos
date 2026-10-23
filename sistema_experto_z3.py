"""
Práctica de Sistemas Expertos y Lógica I:
 Traductor de Lenguaje Natural, Árboles Semánticos y Constantes Z3
=============================================================================
Basado en la metodología de Árboles Semánticos (Tableaux) de la Prof.ª Yolanda
Torres Falcón (Lógica I) y modelado simbólico con constantes booleanas (Z3).

Qué realiza este sistema:
-------------------------
1. Recibe una Base de Conocimiento (varias reglas y hechos) en español.
2. Identifica proposiciones atómicas simples y asigna letras (p, q, r...).
   - Unifica la misma idea en distintas reglas bajo la misma letra atómica.
   - Reconoce la negación («no ...») como ¬átomo sin crear un átomo nuevo.
3. Muestra la Tabla de Átomos (letra y significado).
4. Reconoce los 5 conectivos lógicos:
   - «y»              -> ∧ (Conjunción)
   - «o»              -> ∨ (Disyunción)
   - «no»             -> ¬ (Negación)
   - «si… entonces»   -> → (Implicación / Condicional)
   - «si y solo si»   -> ↔ (Bicondicional / Equivalencia)
5. Genera fórmulas bien formadas (FBF) con paréntesis cuando es necesario y
   muestra cada regla numerada (R1, R2, R3...).
6. Construye el Árbol Semántico suponiendo que todas las reglas son verdaderas:
   - Aplica reglas de 1 rama (α): ∧, ¬∨, ¬→, ¬¬
   - Aplica reglas de 2 ramas (β): ∨, →, ↔, ¬∧, ¬↔
   - Desarrolla cada fórmula en todas las ramas abiertas debajo de ella.
   - Marca con ✓ las fórmulas compuestas ya aplicadas.
   - Cierra con ✕ las ramas que contienen un átomo y su negación (ej. p y ¬p).
   - Termina cuando solo quedan literales (letras o letras negadas).
7. Dibuja visualmente el Árbol Semántico en 2D y reporta los modelos de las
   ramas abiertas (ej. p = V, q = V, r = F) o informa si la base de
   conocimiento es insatisfacible (inconsistente).
8. Genera además la representación equivalente en constantes de Z3 Solver.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from z3 import Bool, And as Z3And, Or as Z3Or, Not as Z3Not, Implies as Z3Implies, Solver, sat, unsat
    Z3_DISPONIBLE = True
except ImportError:
    Z3_DISPONIBLE = False


# =============================================================================
# 1. REPRESENTACIÓN DE FÓRMULAS LÓGICAS (AST)
# =============================================================================

class Formula:
    """Clase base para todas las fórmulas bien formadas (FBF)."""
    def es_compuesta(self) -> bool:
        return True

    def es_literal(self) -> bool:
        return False


@dataclass(frozen=True)
class Atomo(Formula):
    letra: str
    significado: str = ""

    def es_compuesta(self) -> bool:
        return False

    def es_literal(self) -> bool:
        return True

    def __str__(self) -> str:
        return self.letra


@dataclass(frozen=True)
class Negacion(Formula):
    sub: Formula

    def es_compuesta(self) -> bool:
        # ¬p es un literal (letra negada), por lo que ya no se expande.
        # ¬(Fórmula Compuesta) o ¬¬p sí es compuesta y debe expandirse.
        return not isinstance(self.sub, Atomo)

    def es_literal(self) -> bool:
        return isinstance(self.sub, Atomo)

    def __str__(self) -> str:
        if isinstance(self.sub, (Atomo, Negacion)):
            return f"¬{self.sub}"
        return f"¬({self.sub})"


@dataclass(frozen=True)
class Conjuncion(Formula):
    izq: Formula
    der: Formula

    def _fmt(self, f: Formula) -> str:
        if isinstance(f, (Atomo, Negacion)):
            return str(f)
        return f"({f})"

    def __str__(self) -> str:
        return f"{self._fmt(self.izq)} ∧ {self._fmt(self.der)}"


@dataclass(frozen=True)
class Disyuncion(Formula):
    izq: Formula
    der: Formula

    def _fmt(self, f: Formula) -> str:
        if isinstance(f, (Atomo, Negacion)):
            return str(f)
        return f"({f})"

    def __str__(self) -> str:
        return f"{self._fmt(self.izq)} ∨ {self._fmt(self.der)}"


@dataclass(frozen=True)
class Implicacion(Formula):
    izq: Formula
    der: Formula

    def _fmt(self, f: Formula) -> str:
        if isinstance(f, (Atomo, Negacion)):
            return str(f)
        return f"({f})"

    def __str__(self) -> str:
        return f"{self._fmt(self.izq)} → {self._fmt(self.der)}"


@dataclass(frozen=True)
class Bicondicional(Formula):
    izq: Formula
    der: Formula

    def _fmt(self, f: Formula) -> str:
        if isinstance(f, (Atomo, Negacion)):
            return str(f)
        return f"({f})"

    def __str__(self) -> str:
        return f"{self._fmt(self.izq)} ↔ {self._fmt(self.der)}"


# =============================================================================
# 2. ANALIZADOR DE LENGUAJE NATURAL EN ESPAÑOL -> FÓRMULAS Y ÁTOMOS
# =============================================================================

class AnalizadorLenguajeNatural:
    """
    Traduce oraciones en español a fórmulas proposicionales bien formadas,
    identificando proposiciones simples y asignándoles letras (p, q, r...).
    """
    LETRAS_DISPONIBLES = list("pqrstuvwxyzabcdefghijklmno")

    def __init__(self):
        self.clave_a_atomo: Dict[str, Atomo] = {}
        self.letra_a_atomo: Dict[str, Atomo] = {}
        self.orden_atomos: List[Atomo] = []

    @staticmethod
    def _normalizar_clave(texto: str) -> str:
        """
        Normaliza una proposición simple para comparar ideas equivalentes:
        quita espacios extra, convierte a minúsculas y remueve tildes para la clave.
        """
        limpio = texto.strip().lower()
        limpio = re.sub(r"^[¿¡\"'«»\s]+|[?!.\"'«»\s]+$", "", limpio)
        limpio = re.sub(r"\s+", " ", limpio)
        sin_tildes = "".join(
            c for c in unicodedata.normalize("NFD", limpio)
            if unicodedata.category(c) != "Mn"
        )
        return sin_tildes

    @staticmethod
    def _limpiar_texto_mostrar(texto: str) -> str:
        limpio = texto.strip()
        limpio = re.sub(r"^[¿¡\"'«»\s]+|[?!.\"'«»\s]+$", "", limpio)
        limpio = re.sub(r"\s+", " ", limpio)
        if limpio:
            limpio = limpio[0].lower() + limpio[1:]
        return limpio

    def obtener_o_crear_atomo(self, texto_proposicion: str) -> Atomo:
        """
        Devuelve el átomo existente si la proposición ya apareció antes,
        o le asigna la siguiente letra disponible (p, q, r...).
        """
        clave = self._normalizar_clave(texto_proposicion)
        if clave in self.clave_a_atomo:
            return self.clave_a_atomo[clave]

        idx = len(self.orden_atomos)
        if idx < len(self.LETRAS_DISPONIBLES):
            letra = self.LETRAS_DISPONIBLES[idx]
        else:
            letra = f"p{idx + 1}"

        significado = self._limpiar_texto_mostrar(texto_proposicion)
        atomo = Atomo(letra=letra, significado=significado)
        self.clave_a_atomo[clave] = atomo
        self.letra_a_atomo[letra] = atomo
        self.orden_atomos.append(atomo)
        return atomo

    @staticmethod
    def _quitar_parentesis_externos(texto: str) -> str:
        t = texto.strip()
        while t.startswith("(") and t.endswith(")"):
            profundidad = 0
            envuelve_todo = True
            for i, ch in enumerate(t):
                if ch == "(":
                    profundidad += 1
                elif ch == ")":
                    profundidad -= 1
                    if profundidad == 0 and i < len(t) - 1:
                        envuelve_todo = False
                        break
            if envuelve_todo:
                t = t[1:-1].strip()
            else:
                break
        return t

    @staticmethod
    def _buscar_fuera_de_parentesis(texto: str, patron: str) -> Optional[re.Match]:
        """Busca una expresión regular únicamente en el nivel 0 de paréntesis."""
        for m in re.finditer(patron, texto, flags=re.IGNORECASE):
            inicio = m.start()
            prof = 0
            for ch in texto[:inicio]:
                if ch == "(":
                    prof += 1
                elif ch == ")":
                    prof = max(0, prof - 1)
            if prof == 0:
                return m
        return None

    def parsear_oracion(self, oracion: str) -> Formula:
        """Convierte una oración en español en su árbol sintáctico (Formula)."""
        t = oracion.strip()
        t = re.sub(r"^[¿¡\"'«»\s]+|[?!.\"'«»\s]+$", "", t)
        t = self._quitar_parentesis_externos(t)

        # 1. Negación de frase compuesta explícita:
        # Ej: "No es cierto que ...", "Es falso que ...", "No ocurre que ...", "No (...)"
        m_neg_comp = re.match(
            r"^(?:no\s+es\s+cierto\s+que|es\s+falso\s+que|no\s+ocurre\s+que|no\s+pasa\s+que|no\s+se\s+cumple\s+que)\s+(.+)$",
            t, flags=re.IGNORECASE
        )
        if m_neg_comp:
            return Negacion(self.parsear_oracion(m_neg_comp.group(1)))

        if re.match(r"^(?:no|¬)\s*\(.+\)$", t, flags=re.IGNORECASE):
            sub_txt = re.sub(r"^(?:no|¬)\s*", "", t, flags=re.IGNORECASE).strip()
            if self._quitar_parentesis_externos(sub_txt) != sub_txt:
                return Negacion(self.parsear_oracion(sub_txt))

        # 2. Bicondicional («si y solo si» / «si y sólo si») en nivel 0
        m_bicond = self._buscar_fuera_de_parentesis(
            t, r",?\s*\bsi\s+y\s+s[oó]lo\s+si\b\s*"
        )
        if m_bicond:
            izq = t[:m_bicond.start()].strip()
            der = t[m_bicond.end():].strip()
            return Bicondicional(self.parsear_oracion(izq), self.parsear_oracion(der))

        # 3. Condicional («si ... entonces ...» o «si ..., ...»)
        if re.match(r"^si\b", t, flags=re.IGNORECASE):
            cuerpo = re.sub(r"^si\s+", "", t, flags=re.IGNORECASE).strip()
            m_entonces = self._buscar_fuera_de_parentesis(
                cuerpo, r",?\s*\bentonces\b\s*"
            )
            if m_entonces:
                ant = cuerpo[:m_entonces.start()].strip(" ,")
                cons = cuerpo[m_entonces.end():].strip(" ,")
                return Implicacion(self.parsear_oracion(ant), self.parsear_oracion(cons))
            m_coma = self._buscar_fuera_de_parentesis(cuerpo, r",\s*")
            if m_coma:
                ant = cuerpo[:m_coma.start()].strip()
                cons = cuerpo[m_coma.end()].strip()
                return Implicacion(self.parsear_oracion(ant), self.parsear_oracion(cons))

        # Condicional sin "si" inicial pero con "entonces" o "implica que"
        m_impl = self._buscar_fuera_de_parentesis(
            t, r",?\s*\b(?:entonces|implica\s+que|por\s+lo\s+tanto)\b\s*"
        )
        if m_impl:
            ant = t[:m_impl.start()].strip(" ,")
            cons = t[m_impl.end()].strip(" ,")
            return Implicacion(self.parsear_oracion(ant), self.parsear_oracion(cons))

        # Condicional invertido: «A si B» (cuando no es «si y solo si»)
        m_si_inv = self._buscar_fuera_de_parentesis(t, r",?\s+\bsi\s+")
        if m_si_inv:
            cons = t[:m_si_inv.start()].strip(" ,")
            ant = t[m_si_inv.end()].strip(" ,")
            return Implicacion(self.parsear_oracion(ant), self.parsear_oracion(cons))

        # 4. Disyunción («o» / «o bien ... o ...») en nivel 0
        t_disy = re.sub(r"^o\s+bien\s+", "", t, flags=re.IGNORECASE).strip()
        m_disy = self._buscar_fuera_de_parentesis(
            t_disy, r",?\s+\b(?:o\s+bien|o|u)\b\s+"
        )
        if m_disy:
            izq = t_disy[:m_disy.start()].strip(" ,")
            der = t_disy[m_disy.end():].strip(" ,")
            return Disyuncion(self.parsear_oracion(izq), self.parsear_oracion(der))

        # 5. Expresión «ni A ni B» -> ¬(A ∨ B)
        m_ni = re.match(r"^ni\s+(.+?)\s+ni\s+(.+)$", t, flags=re.IGNORECASE)
        if m_ni:
            return Negacion(
                Disyuncion(
                    self.parsear_oracion(m_ni.group(1)),
                    self.parsear_oracion(m_ni.group(2))
                )
            )

        # 6. Conjunción («y» / «e» / «pero» / «además») en nivel 0
        m_conj = self._buscar_fuera_de_parentesis(
            t, r",?\s+\b(?:y|e|pero|adem[aá]s)\b\s+"
        )
        if m_conj:
            izq = t[:m_conj.start()].strip(" ,")
            der = t[m_conj.end():].strip(" ,")
            return Conjuncion(self.parsear_oracion(izq), self.parsear_oracion(der))

        # 7. Negación en proposición simple («no ...» o «¬...»)
        m_no_inicio = re.match(r"^(?:no\s+|¬\s*)(.+)$", t, flags=re.IGNORECASE)
        if m_no_inicio:
            resto = m_no_inicio.group(1).strip()
            return Negacion(self.parsear_oracion(resto))

        # Negación interna antes del verbo (ej. "el tren no llega" -> ¬("el tren llega"))
        m_no_interno = re.match(r"^(.+?)\s+no\s+(.+)$", t, flags=re.IGNORECASE)
        if m_no_interno:
            frase_positiva = f"{m_no_interno.group(1).strip()} {m_no_interno.group(2).strip()}"
            return Negacion(self.obtener_o_crear_atomo(frase_positiva))

        # 8. Proposición atómica simple positiva
        return self.obtener_o_crear_atomo(t)


# =============================================================================
# 3. MOTOR DE ÁRBOLES SEMÁNTICOS (TABLEAUX PROPOSICIONAL)
# =============================================================================

@dataclass
class ItemFormula:
    """Representa una fórmula colocada en un nodo del árbol y si ya fue aplicada."""
    formula: Formula
    etiqueta: str = ""
    aplicada: bool = False

    def texto_mostrar(self) -> str:
        marca = " ✓" if self.aplicada else ""
        prefijo = f"{self.etiqueta}: " if self.etiqueta else ""
        return f"{prefijo}{self.formula}{marca}"


@dataclass
class NodoArbol:
    """Nodo del Árbol Semántico."""
    items: List[ItemFormula] = field(default_factory=list)
    hijos: List["NodoArbol"] = field(default_factory=list)
    padre: Optional["NodoArbol"] = None
    cerrada: bool = False
    motivo_cierre: str = ""
    regla_origen: str = ""

    def es_hoja(self) -> bool:
        return len(self.hijos) == 0


class ConstructorArbolSemantico:
    """
    Construye el árbol semántico siguiendo las reglas de expansión α (1 rama)
    y β (2 ramas) de Lógica I (Prof.ª Yolanda Torres Falcón).
    """

    def __init__(self, reglas: List[Tuple[str, Formula]], orden_atomos: List[Atomo]):
        self.reglas_iniciales = reglas
        self.orden_atomos = orden_atomos
        self.pasos_ejecucion: List[str] = []
        self.raiz = NodoArbol(
            items=[ItemFormula(formula=f, etiqueta=etq) for etq, f in reglas]
        )

    @staticmethod
    def clasificar_formula(f: Formula) -> Tuple[str, str, List[List[Formula]]]:
        """
        Clasifica una fórmula compuesta según las 9 reglas de expansión:
        Retorna (tipo_regla, nombre_regla, ramas_generadas)
        donde:
          - tipo_regla == 'ALFA' (1 rama: ∧, ¬∨, ¬→, ¬¬)
          - tipo_regla == 'BETA' (2 ramas: ∨, →, ↔, ¬∧, ¬↔)
        """
        if isinstance(f, Conjuncion):
            return ("ALFA", "Conjunción (∧)", [[f.izq, f.der]])

        if isinstance(f, Disyuncion):
            return ("BETA", "Disyunción (∨)", [[f.izq], [f.der]])

        if isinstance(f, Implicacion):
            return ("BETA", "Implicación (→)", [[Negacion(f.izq)], [f.der]])

        if isinstance(f, Bicondicional):
            return ("BETA", "Bicondicional (↔)", [[f.izq, f.der], [Negacion(f.izq), Negacion(f.der)]])

        if isinstance(f, Negacion):
            sub = f.sub
            if isinstance(sub, Negacion):
                return ("ALFA", "Doble negación (¬¬)", [[sub.sub]])

            if isinstance(sub, Disyuncion):
                return ("ALFA", "Negación de disyunción (¬∨)", [[Negacion(sub.izq), Negacion(sub.der)]])

            if isinstance(sub, Implicacion):
                return ("ALFA", "Negación de implicación (¬→)", [[sub.izq, Negacion(sub.der)]])

            if isinstance(sub, Conjuncion):
                return ("BETA", "Negación de conjunción (¬∧)", [[Negacion(sub.izq)], [Negacion(sub.der)]])

            if isinstance(sub, Bicondicional):
                return (
                    "BETA",
                    "Negación de bicondicional (¬↔)",
                    [[sub.izq, Negacion(sub.der)], [Negacion(sub.izq), sub.der]]
                )

        raise ValueError(f"La fórmula '{f}' no es una fórmula compuesta expandible.")

    @staticmethod
    def obtener_literales_en_rama(hoja: NodoArbol) -> Tuple[Dict[str, bool], Optional[str]]:
        """
        Recorre desde la raíz hasta la hoja recolectando los literales (p, ¬p).
        Si detecta un átomo y su negación en la misma rama, devuelve el motivo de contradicción.
        """
        ancestros: List[NodoArbol] = []
        actual: Optional[NodoArbol] = hoja
        while actual is not None:
            ancestros.append(actual)
            actual = actual.padre
        ancestros.reverse()

        positivos = set()
        negativos = set()
        contradiccion = None

        for nodo in ancestros:
            for item in nodo.items:
                f = item.formula
                if isinstance(f, Atomo):
                    positivos.add(f.letra)
                    if f.letra in negativos and contradiccion is None:
                        contradiccion = f"{f.letra} y ¬{f.letra}"
                elif isinstance(f, Negacion) and isinstance(f.sub, Atomo):
                    letra = f.sub.letra
                    negativos.add(letra)
                    if letra in positivos and contradiccion is None:
                        contradiccion = f"{letra} y ¬{letra}"

        asignacion: Dict[str, bool] = {}
        for l in positivos:
            asignacion[l] = True
        for l in negativos:
            if l not in asignacion:
                asignacion[l] = False

        return asignacion, contradiccion

    def _actualizar_cierre(self, hoja: NodoArbol):
        _, contradiccion = self.obtener_literales_en_rama(hoja)
        if contradiccion is not None:
            hoja.cerrada = True
            hoja.motivo_cierre = contradiccion

    def _hojas_abiertas_en_subarbol(self, nodo: NodoArbol) -> List[NodoArbol]:
        if nodo.es_hoja():
            return [] if nodo.cerrada else [nodo]
        abiertas: List[NodoArbol] = []
        for h in nodo.hijos:
            abiertas.extend(self._hojas_abiertas_en_subarbol(h))
        return abiertas

    def _buscar_siguiente_formula_a_expandir(self) -> Optional[Tuple[NodoArbol, ItemFormula]]:
        """
        Busca la siguiente fórmula compuesta no aplicada que tenga al menos
        una rama abierta debajo.
        Prioriza las reglas ALFA (1 rama) sobre las BETA (2 ramas), y entre
        las BETA prioriza aquellas que afectan a menos hojas abiertas para
        cerrar ramas contradictorias antes de duplicar subárboles.
        """
        candidatos_alfa: List[Tuple[int, int, NodoArbol, ItemFormula]] = []
        candidatos_beta: List[Tuple[int, int, NodoArbol, ItemFormula]] = []

        orden = 0
        cola: List[NodoArbol] = [self.raiz]
        while cola:
            nodo = cola.pop(0)
            hojas_abiertas = self._hojas_abiertas_en_subarbol(nodo)
            if not hojas_abiertas:
                continue

            for item in nodo.items:
                if item.formula.es_compuesta() and not item.aplicada:
                    tipo, _, _ = self.clasificar_formula(item.formula)
                    if tipo == "ALFA":
                        candidatos_alfa.append((len(hojas_abiertas), orden, nodo, item))
                    else:
                        candidatos_beta.append((len(hojas_abiertas), orden, nodo, item))
                    orden += 1

            for h in nodo.hijos:
                cola.append(h)

        if candidatos_alfa:
            candidatos_alfa.sort(key=lambda x: (x[0], x[1]))
            _, _, n, it = candidatos_alfa[0]
            return n, it
        if candidatos_beta:
            candidatos_beta.sort(key=lambda x: (x[0], x[1]))
            _, _, n, it = candidatos_beta[0]
            return n, it
        return None

    def construir(self):
        """Desarrolla el árbol semántico hasta que solo queden literales o ramas cerradas."""
        self._actualizar_cierre(self.raiz)
        if self.raiz.cerrada:
            self.pasos_ejecucion.append(
                f"La raíz se cierra inmediatamente por contradicción directa ({self.raiz.motivo_cierre})."
            )
            return

        num_paso = 1
        while True:
            seleccion = self._buscar_siguiente_formula_a_expandir()
            if seleccion is None:
                break

            nodo_origen, item = seleccion
            tipo, nombre_regla, ramas_generadas = self.clasificar_formula(item.formula)
            item.aplicada = True

            hojas_destino = self._hojas_abiertas_en_subarbol(nodo_origen)
            desc_ramas = " y ".join(
                "[" + ", ".join(str(x) for x in r) + "]" for r in ramas_generadas
            )
            self.pasos_ejecucion.append(
                f"Paso {num_paso}: Se aplica {nombre_regla} a '{item.formula}' ✓ "
                f"en {len(hojas_destino)} rama(s) abierta(s) -> genera {desc_ramas}."
            )
            num_paso += 1

            for hoja in hojas_destino:
                if tipo == "ALFA":
                    nuevos_items = [ItemFormula(formula=f) for f in ramas_generadas[0]]
                    hijo = NodoArbol(
                        items=nuevos_items,
                        padre=hoja,
                        regla_origen=nombre_regla
                    )
                    hoja.hijos = [hijo]
                    self._actualizar_cierre(hijo)
                else:
                    items_izq = [ItemFormula(formula=f) for f in ramas_generadas[0]]
                    items_der = [ItemFormula(formula=f) for f in ramas_generadas[1]]
                    hijo_izq = NodoArbol(
                        items=items_izq,
                        padre=hoja,
                        regla_origen=nombre_regla
                    )
                    hijo_der = NodoArbol(
                        items=items_der,
                        padre=hoja,
                        regla_origen=nombre_regla
                    )
                    hoja.hijos = [hijo_izq, hijo_der]
                    self._actualizar_cierre(hijo_izq)
                    self._actualizar_cierre(hijo_der)

    def obtener_todas_las_hojas(self) -> List[NodoArbol]:
        hojas: List[NodoArbol] = []

        def _rec(n: NodoArbol):
            if n.es_hoja():
                hojas.append(n)
            else:
                for h in n.hijos:
                    _rec(h)

        _rec(self.raiz)
        return hojas

    # -------------------------------------------------------------------------
    # VISUALIZACIÓN 2D DEL ÁRBOL SEMÁNTICO EN CONSOLA
    # -------------------------------------------------------------------------
    def renderizar_arbol_2d(self) -> str:
        """
        Genera una representación visual en 2D del árbol semántico con ramas,
        marcas ✓ en fórmulas aplicadas, ✕ en ramas cerradas y ◯ en ramas abiertas.
        """
        def _render_nodo(nodo: NodoArbol) -> Tuple[List[str], int, int]:
            lineas_nodo = [it.texto_mostrar() for it in nodo.items]
            if nodo.es_hoja():
                if nodo.cerrada:
                    lineas_nodo.append(f"✕ ({nodo.motivo_cierre})")
                else:
                    lineas_nodo.append("◯ (Abierta)")

            ancho_propio = max((len(s) for s in lineas_nodo), default=1)
            bloque_propio = [s.center(ancho_propio) for s in lineas_nodo]
            mid_propio = ancho_propio // 2

            # Caso Hoja
            if nodo.es_hoja():
                return bloque_propio, ancho_propio, mid_propio

            # Caso 1 Hijo (Regla ALFA: 1 rama vertical │)
            if len(nodo.hijos) == 1:
                h_lineas, h_ancho, h_mid = _render_nodo(nodo.hijos[0])
                mid = max(mid_propio, h_mid)
                pad_arriba = mid - mid_propio
                pad_abajo = mid - h_mid
                ancho_total = max(pad_arriba + ancho_propio, pad_abajo + h_ancho)

                arriba = [(" " * pad_arriba + s).ljust(ancho_total) for s in bloque_propio]
                conector = [(" " * mid + "│").ljust(ancho_total)]
                abajo = [(" " * pad_abajo + l).ljust(ancho_total) for l in h_lineas]
                return arriba + conector + abajo, ancho_total, mid

            # Caso 2 Hijos (Regla BETA: bifurcación ┌───┴───┐)
            izq_l, izq_w, izq_mid = _render_nodo(nodo.hijos[0])
            der_l, der_w, der_mid = _render_nodo(nodo.hijos[1])

            separacion = 6
            ancho_hijos = izq_w + separacion + der_w
            pos_izq = izq_mid
            pos_der = izq_w + separacion + der_mid
            centro_rama = (pos_izq + pos_der) // 2

            mid = max(mid_propio, centro_rama)
            pad_arriba = mid - mid_propio
            offset_hijos = mid - centro_rama

            pos_izq += offset_hijos
            pos_der += offset_hijos
            centro_rama += offset_hijos

            ancho_total = max(pad_arriba + ancho_propio, offset_hijos + ancho_hijos)
            arriba = [(" " * pad_arriba + s).ljust(ancho_total) for s in bloque_propio]

            # Construir línea de ramificación ┌───┴───┐
            chars_rama = [" "] * ancho_total
            chars_rama[pos_izq] = "┌"
            chars_rama[pos_der] = "┐"
            for i in range(pos_izq + 1, pos_der):
                chars_rama[i] = "─"
            chars_rama[centro_rama] = "┴"
            linea_rama = "".join(chars_rama)

            altura_max = max(len(izq_l), len(der_l))
            izq_pad = izq_l + [" " * izq_w] * (altura_max - len(izq_l))
            der_pad = der_l + [" " * der_w] * (altura_max - len(der_l))

            abajo = []
            for l_i, l_d in zip(izq_pad, der_pad):
                fila = (" " * offset_hijos) + l_i + (" " * separacion) + l_d
                abajo.append(fila.ljust(ancho_total))

            return arriba + [linea_rama] + abajo, ancho_total, mid

        lineas, _, _ = _render_nodo(self.raiz)
        return "\n".join("  " + l.rstrip() for l in lineas)


# =============================================================================
# 4. INTEGRACIÓN Y REPORTE COMPLETO DEL SISTEMA DE REGLAS
# =============================================================================

class SistemaReglasSemantico:
    """
    Integra el analizador en español, la tabla de átomos, el árbol semántico
    y la conversión opcional a constantes de Z3 Solver.
    """

    def __init__(self, titulo: str = "Evaluación de Base de Conocimiento"):
        self.titulo = titulo
        self.analizador = AnalizadorLenguajeNatural()
        self.oraciones_originales: List[str] = []
        self.reglas: List[Tuple[str, Formula]] = []

    def cargar_reglas(self, oraciones: List[str]):
        """Recibe una lista de reglas/hechos en español y construye la BC."""
        for idx, texto in enumerate(oraciones, start=1):
            if not texto.strip():
                continue
            self.oraciones_originales.append(texto.strip())
            formula = self.analizador.parsear_oracion(texto)
            etiqueta = f"R{idx}"
            self.reglas.append((etiqueta, formula))

    def ejecutar_y_mostrar(self):
        print("\n" + "═" * 74)
        print(f" {self.titulo.upper()}")
        print("═" * 74)

        # 1. Entrada en lenguaje natural
        print("\n1. REGLAS Y HECHOS EN LENGUAJE NATURAL (ENTRADA):")
        for idx, oracion in enumerate(self.oraciones_originales, start=1):
            print(f"   • Regla {idx}: «{oracion}»")

        # 2. Tabla de átomos (proposiciones simples)
        print("\n2. TABLA DE ÁTOMOS (PROPOSICIONES SIMPLES):")
        print(f"   ┌─────────┬────────────────────────────────────────────────────────┐")
        print(f"   │ Símbolo │ Significado (Proposición Simple)                       │")
        print(f"   ├─────────┼────────────────────────────────────────────────────────┤")
        for atomo in self.analizador.orden_atomos:
            print(f"   │    {atomo.letra:<4} │ {atomo.significado:<54} │")
        print(f"   └─────────┴────────────────────────────────────────────────────────┘")

        # 3. Fórmulas bien formadas
        print("\n3. TRADUCCIÓN A FÓRMULAS LÓGICAS BIEN FORMADAS:")
        for etiqueta, formula in self.reglas:
            print(f"   • {etiqueta}: {formula}")

        # 4. Construcción y visualización del Árbol Semántico
        constructor = ConstructorArbolSemantico(
            self.reglas, self.analizador.orden_atomos
        )
        constructor.construir()

        print("\n4. APLICACIÓN DE REGLAS DE EXPANSIÓN:")
        if constructor.pasos_ejecucion:
            for paso in constructor.pasos_ejecucion:
                print(f"   - {paso}")
        else:
            print("   - Todas las fórmulas iniciales ya son literales simples.")

        print("\n5. VISUALIZACIÓN DEL ÁRBOL SEMÁNTICO:")
        print("   (Leyenda: ✓ = fórmula aplicada | ✕ = rama cerrada | ◯ = rama abierta)\n")
        print(constructor.renderizar_arbol_2d())

        # 6. Análisis de ramas y extracción de modelos
        hojas = constructor.obtener_todas_las_hojas()
        cerradas = [h for h in hojas if h.cerrada]
        abiertas = [h for h in hojas if not h.cerrada]

        print("\n6. RESULTADO Y CONDICIONES DE CUMPLIMIENTO (MODELOS):")
        print(
            f"   • Resumen de ramas: {len(cerradas)} cerrada(s) (✕) y "
            f"{len(abiertas)} abierta(s) (◯)."
        )

        if not abiertas:
            print(
                "   • CONCLUSIÓN: La base de conocimiento es INSATISFACIBLE (inconsistente).\n"
                "     Todas las ramas se cerraron (✕); no existe ninguna asignación de valores\n"
                "     de verdad en la que todas las reglas se cumplan al mismo tiempo."
            )
        else:
            print(
                "   • CONCLUSIÓN: La base de conocimiento es SATISFACIBLE (consistente).\n"
                "     Todas las reglas se cumplen simultáneamente bajo las siguientes\n"
                "     asignaciones de verdad (ramas abiertas):"
            )
            for idx, hoja in enumerate(abiertas, start=1):
                asignacion, _ = constructor.obtener_literales_en_rama(hoja)
                partes_modelo = []
                detalles_significado = []
                for atomo in self.analizador.orden_atomos:
                    if atomo.letra in asignacion:
                        val_str = "V" if asignacion[atomo.letra] else "F"
                        partes_modelo.append(f"{atomo.letra} = {val_str}")
                        estado_nl = (
                            f"sí «{atomo.significado}»"
                            if asignacion[atomo.letra]
                            else f"no «{atomo.significado}»"
                        )
                        detalles_significado.append(estado_nl)
                    else:
                        partes_modelo.append(f"{atomo.letra} = V/F (libre)")
                modelo_txt = ", ".join(partes_modelo)
                print(f"     - Rama abierta {idx}: {modelo_txt}")
                print(f"       Interpretación: {', '.join(detalles_significado)}.")

        # 7. Equivalencia con Constantes de Z3 Solver
        self._mostrar_equivalencia_z3()
        print("═" * 74 + "\n")

    def _a_expresion_z3_texto(self, f: Formula) -> str:
        if isinstance(f, Atomo):
            return f.letra
        if isinstance(f, Negacion):
            return f"Not({self._a_expresion_z3_texto(f.sub)})"
        if isinstance(f, Conjuncion):
            return f"And({self._a_expresion_z3_texto(f.izq)}, {self._a_expresion_z3_texto(f.der)})"
        if isinstance(f, Disyuncion):
            return f"Or({self._a_expresion_z3_texto(f.izq)}, {self._a_expresion_z3_texto(f.der)})"
        if isinstance(f, Implicacion):
            return f"Implies({self._a_expresion_z3_texto(f.izq)}, {self._a_expresion_z3_texto(f.der)})"
        if isinstance(f, Bicondicional):
            return f"({self._a_expresion_z3_texto(f.izq)} == {self._a_expresion_z3_texto(f.der)})"
        return str(f)

    def _mostrar_equivalencia_z3(self):
        print("\n7. REPRESENTACIÓN EN CONSTANTES DE Z3 SOLVER:")
        letras = [a.letra for a in self.analizador.orden_atomos]
        for a in self.analizador.orden_atomos:
            print(f"   {a.letra} = Bool('{a.letra}')  # Constante booleana: «{a.significado}»")
        print("   s = Solver()")
        for etq, f in self.reglas:
            print(f"   s.add({self._a_expresion_z3_texto(f)})  # {etq}: {f}")

        if Z3_DISPONIBLE:
            constantes_z3 = {a.letra: Bool(a.letra) for a in self.analizador.orden_atomos}

            def _construir_z3(f: Formula):
                if isinstance(f, Atomo):
                    return constantes_z3[f.letra]
                if isinstance(f, Negacion):
                    return Z3Not(_construir_z3(f.sub))
                if isinstance(f, Conjuncion):
                    return Z3And(_construir_z3(f.izq), _construir_z3(f.der))
                if isinstance(f, Disyuncion):
                    return Z3Or(_construir_z3(f.izq), _construir_z3(f.der))
                if isinstance(f, Implicacion):
                    return Z3Implies(_construir_z3(f.izq), _construir_z3(f.der))
                if isinstance(f, Bicondicional):
                    return _construir_z3(f.izq) == _construir_z3(f.der)

            s = Solver()
            for _, f in self.reglas:
                s.add(_construir_z3(f))
            if s.check() == sat:
                m = s.model()
                vals = [
                    f"{l} = {'V' if str(m.eval(constantes_z3[l], model_completion=True)) == 'True' else 'F'}"
                    for l in letras
                ]
                print(f"   -> Verificación Z3 (sat): {', '.join(vals)}")
            else:
                print("   -> Verificación Z3 (unsat): Base de conocimiento insatisfacible.")


# =============================================================================
# 5. CASOS DE PRUEBA Y MODO INTERACTIVO
# =============================================================================

def ejecutar_caso_minimo_video():
    """
    Caso de prueba mínimo requerido por la rúbrica:
    - «Si es de día, entonces voy a trabajar»
    - «Si llueve, entonces no voy a trabajar»
    - «Es de día»
    Resultado esperado: 2 ramas cerradas y 1 abierta con el modelo p = V, q = V, r = F.
    """
    sistema = SistemaReglasSemantico("Caso de Prueba Mínimo (Ejemplo del Video)")
    sistema.cargar_reglas([
        "Si es de día, entonces voy a trabajar",
        "Si llueve, entonces no voy a trabajar",
        "Es de día"
    ])
    sistema.ejecutar_y_mostrar()


def ejecutar_caso_cinco_conectivos():
    """
    Caso de prueba que ejercita conectivos compuestos (∧, ∨, ¬, →, ↔) y
    reglas de 1 rama (α) y 2 ramas (β).
    """
    sistema = SistemaReglasSemantico("Caso de Prueba 2: Conectivos Compuestos (∧, ∨, ¬, →, ↔)")
    sistema.cargar_reglas([
        "Si hay energía y el servidor responde, entonces el sistema opera",
        "El sistema opera si y solo si hay respaldo",
        "No ocurre que (no hay energía o haya falla de red)",
        "El servidor responde"
    ])
    sistema.ejecutar_y_mostrar()


def ejecutar_caso_insatisfacible():
    """
    Caso de prueba donde todas las ramas se cierran (✕), demostrando que
    la base de conocimiento es insatisfacible (inconsistente).
    """
    sistema = SistemaReglasSemantico("Caso de Prueba 3: Base de Conocimiento Insatisfacible")
    sistema.cargar_reglas([
        "Si es de día, entonces voy a trabajar",
        "Si llueve, entonces no voy a trabajar",
        "Es de día y llueve"
    ])
    sistema.ejecutar_y_mostrar()


def modo_interactivo():
    """Permite al usuario ingresar sus propias reglas en español desde teclado."""
    print("\n" + "─" * 74)
    print(" MODO INTERACTIVO: INGRESA TUS REGLAS EN ESPAÑOL")
    print(" (Escribe una regla o hecho por línea; presiona ENTER en vacío para terminar)")
    print("─" * 74)
    reglas_usuario = []
    while True:
        try:
            linea = input(f"Regla {len(reglas_usuario) + 1}: ").strip()
        except EOFError:
            break
        if not linea:
            break
        reglas_usuario.append(linea)

    if reglas_usuario:
        sistema = SistemaReglasSemantico("Base de Conocimiento del Usuario")
        sistema.cargar_reglas(reglas_usuario)
        sistema.ejecutar_y_mostrar()


if __name__ == "__main__":
    import sys

    # 1. Ejecutar el caso de prueba mínimo obligatorio
    ejecutar_caso_minimo_video()

    # 2. Ejecutar casos demostrativos adicionales (conectivos completos e insatisfacibilidad)
    ejecutar_caso_cinco_conectivos()
    ejecutar_caso_insatisfacible()

    # 3. Si se ejecuta con '--interactivo', permite escribir nuevas reglas en consola
    if "--interactivo" in sys.argv:
        modo_interactivo()