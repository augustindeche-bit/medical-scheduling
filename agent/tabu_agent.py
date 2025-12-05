

import numpy as np
from typing import Dict, List, Tuple



def decode_chromosome(chromosome, competence_matrix):
    """
    Décodage :
    - on lit les gènes dans l'ordre,
    - on ne planifie (p, op_idx) que si op_idx est la prochaine op pour ce patient,
    - dès qu'on planifie une opération, on repart du début du chromosome,
    - chaque bloc de skill_time est posé de manière consécutive sur le skill.

    Args:
        chromosome: Liste de [patient, op_idx] représentant l'ordre des opérations
        competence_matrix: Matrice des compétences [patient][operation][skill]

    Returns:
        solution: Matrice [skill][time] avec (patient, op_idx) ou None
    """
    nb_patient = len(competence_matrix)
    nb_skills = len(competence_matrix[0][0])

    solution = [[] for _ in range(nb_skills)]
    next_op = [0 for _ in range(nb_patient)]
    patient_end = [0 for _ in range(nb_patient)]

    total_ops = sum(len(competence_matrix[p]) for p in range(nb_patient))
    scheduled_ops = 0

    while scheduled_ops < total_ops:
        progress = False

        for patient, op_idx in chromosome:

            if op_idx < next_op[patient]:
                continue
            if op_idx > next_op[patient]:
                continue

            operation = competence_matrix[patient][op_idx]
            last_time_for_operation = patient_end[patient]

            for skill in range(nb_skills):
                skill_time = operation[skill]
                if skill_time <= 0:
                    continue

                t = patient_end[patient]

                while True:
                    current_len = len(solution[0]) if solution[0] else 0
                    while t + skill_time > current_len:
                        if not solution[0]:
                            for skill_row in solution:
                                skill_row.append(None)
                        else:
                            for skill_row in solution:
                                skill_row.append(None)
                        current_len += 1

                    bloc_libre = True
                    for tau in range(t, t + skill_time):
                        if solution[skill][tau] is not None:
                            bloc_libre = False
                            break

                    if bloc_libre:
                        for tau in range(t, t + skill_time):
                            solution[skill][tau] = (patient, op_idx)
                        fin_bloc = t + skill_time - 1
                        if fin_bloc > last_time_for_operation:
                            last_time_for_operation = fin_bloc
                        break
                    else:
                        t += 1

            patient_end[patient] = last_time_for_operation + 1
            next_op[patient] += 1
            scheduled_ops += 1
            progress = True
            break

        if not progress:
            # En cas de problème, retourne une solution vide
            return solution

    return solution


def calculate_makespan(solution):
    """
    Calcule le makespan (CMax) d'une solution décodée.

    Args:
        solution: Matrice [skill][time] avec (patient, op_idx) ou None

    Returns:
        int: Le makespan (temps total de la planification)
    """
    if not solution or not solution[0]:
        return 0
    return len(solution[0])

# ----------------------- Utilitaires de conversion -----------------------
def normalize_competence_matrix(competence_matrix: List[List[List[int]]]) -> np.ndarray:
    """Convertit une matrice de compétences irrégulière en tableau numpy régulier."""
    num_patients = len(competence_matrix)
    max_operations = max(len(patient_ops) for patient_ops in competence_matrix)
    num_skills = len(competence_matrix[0][0])

    normalized = np.zeros((num_patients, max_operations, num_skills), dtype=int)

    for p in range(num_patients):
        for o in range(len(competence_matrix[p])):
            for s in range(num_skills):
                normalized[p, o, s] = competence_matrix[p][o][s]

    return normalized


def build_tasks_by_comp(C: np.ndarray) -> Dict[int, List[Tuple[str, int, int, int]]]:
    """Construit pour chaque compétence k la liste des tâches unitaires."""
    P, O, K = C.shape
    by_comp: Dict[int, List[Tuple[str, int, int, int]]] = {k: [] for k in range(K)}
    for p in range(P):
        for o in range(O):
            for k in range(K):
                cnt = int(C[p, o, k])
                for u in range(cnt):
                    tid = f"p{p}_o{o}_k{k}_u{u}"
                    by_comp[k].append((tid, p, o, k))
    return by_comp


def parse_tid(tid: str):
    """Parse un task_id de format 'p{p}_o{o}_k{k}_u{u}'"""
    parts = tid.split("_")
    p = int(parts[0][1:])
    o = int(parts[1][1:])
    k = int(parts[2][1:])
    u = int(parts[3][1:])
    return p, o, k, u


def evaluate_with_logs(schedule: Dict[int, List[Tuple[str, int, int, int]]], C: np.ndarray):
    """Simule un planning et retourne (cmax, logs)."""
    P, O, K = C.shape
    queues = {k: lst[:] for k, lst in schedule.items()}
    running: Dict[int, Tuple[str, int]] = {k: None for k in range(K)}
    remaining = {(p, o): int(C[p, o].sum()) for p in range(P) for o in range(O)}

    next_op = {}
    for p in range(P):
        o = 0
        while o < O and remaining[(p, o)] == 0:
            o += 1
        next_op[p] = o

    logs = []
    t = 0
    total_tasks = sum(remaining.values())
    done = 0

    while done < total_tasks:
        for k in range(K):
            if running[k] is not None:
                continue
            q = queues[k]
            chosen = None
            for idx, (tid, p, o, _k) in enumerate(q):
                if next_op[p] == o:
                    chosen = idx
                    break
            if chosen is not None:
                tid, p, o, _k = q[chosen]
                del q[chosen]
                running[k] = (tid, t + 1)
                logs.append({"comp": k, "patient": p, "op": o,
                           "task_id": tid, "start_tick": t, "end_tick": t + 1})

        if all(r is None for r in running.values()):
            raise RuntimeError("Blocage : aucune tâche en cours ni éligible.")

        t += 1

        for k in range(K):
            r = running[k]
            if r is None:
                continue
            tid, end_tick = r
            if end_tick == t:
                running[k] = None
                p, o, _k, _u = parse_tid(tid)
                remaining[(p, o)] -= 1
                done += 1
                if remaining[(p, o)] == 0 and next_op[p] == o:
                    o2 = o + 1
                    while o2 < O and remaining[(p, o2)] == 0:
                        o2 += 1
                    next_op[p] = o2

    if not logs:
        return 0, logs
    cmax_reel = max(log["end_tick"] for log in logs)
    return cmax_reel, logs


def eval_cmax(schedule: Dict[int, List[Tuple[str, int, int, int]]], C: np.ndarray) -> int:
    """Calcule uniquement le makespan."""
    cmax, _ = evaluate_with_logs(schedule, C)
    return cmax


def apply_insertion(schedule: Dict[int, List[Tuple[str, int, int, int]]],
                   k: int, i: int, j: int) -> Dict[int, List[Tuple[str, int, int, int]]]:
    """Insère l'élément i à la position j dans la file de la compétence k."""
    new = {kk: lst[:] for kk, lst in schedule.items()}
    if len(new[k]) < 2 or i == j:
        return new
    i = max(0, min(i, len(new[k]) - 1))
    j = max(0, min(j, len(new[k])))
    elem = new[k].pop(i)
    new[k].insert(j, elem)
    return new


def algo_tabu_step(current_schedule, best_schedule, best_val, tabu_dict,
                   iteration, C_array, tenure=7, candidate_size=40):
    """
    Une seule itération de la recherche tabu.

    Args:
        current_schedule: planning courant
        best_schedule: meilleur planning trouvé
        best_val: makespan du meilleur planning
        tabu_dict: dictionnaire des mouvements tabu {(k, moved_id, j): iteration}
        iteration: numéro d'itération actuel
        C_array: matrice de compétences normalisée
        tenure: durée tabu
        candidate_size: nombre de candidats à générer

    Returns:
        tuple: (nouveau_current, nouveau_best, nouveau_best_val, nouveau_tabu_dict)
    """
    P, O, K = C_array.shape
    candidates = []

    # Générer des insertions aléatoires
    for _ in range(candidate_size):
        k = random.randrange(K)
        if len(current_schedule[k]) < 2:
            continue
        i = random.randrange(len(current_schedule[k]))
        j = random.randrange(len(current_schedule[k]))
        if i == j:
            continue

        moved_id = current_schedule[k][i][0]
        attr = (k, moved_id, j)
        cand = apply_insertion(current_schedule, k, i, j)
        val = eval_cmax(cand, C_array)
        candidates.append((val, cand, attr))

    if not candidates:
        return current_schedule, best_schedule, best_val, tabu_dict

    candidates.sort(key=lambda x: x[0])

    # Choisir le meilleur candidat non-tabu (ou avec aspiration)
    chosen = None
    for val, cand, attr in candidates:
        is_tabu = (attr in tabu_dict) and (tabu_dict[attr] > iteration)
        if (not is_tabu) or (val < best_val):  # aspiration
            chosen = (val, cand, attr)
            break

    if chosen is None:
        chosen = candidates[0]

    val, cand, attr = chosen
    current_schedule = cand
    tabu_dict[attr] = iteration + tenure

    # Mise à jour du meilleur
    if val < best_val:
        best_schedule = cand
        best_val = val

    return current_schedule, best_schedule, best_val, tabu_dict


class TabuAgent(Agent):

    def tabu_search_step(self):
        """Recherche tabu - une itération"""

        # Appel à l'algorithme tabu (une seule itération)
        self.current_schedule, self.best_schedule, self.makespan, self.tabu_dict = algo_tabu_step(
            self.current_schedule,
            self.best_schedule,
            self.makespan,
            self.tabu_dict,
            self.iteration,
            self.model.C_array,
            tenure=self.tenure,
            candidate_size=self.candidate_size
        )

        self.iteration += 1

    def __init__(self, model, collaboratif=False, tenure=7, candidate_size=40):
        super().__init__(model)

        # Initialiser le planning (tâches par compétence)
        self.current_schedule = build_tasks_by_comp(self.model.C_array)
        self.best_schedule = {k: lst[:] for k, lst in self.current_schedule.items()}

        # Calculer le makespan initial
        self.makespan = eval_cmax(self.best_schedule, self.model.C_array)

        # Paramètres de la recherche tabu
        self.tabu_dict = {}
        self.iteration = 1
        self.tenure = tenure
        self.candidate_size = candidate_size

        self.collaboratif = collaboratif

    def contact(self):
        """
        Si collaboratif, récupère le meilleur planning parmi tous les agents
        """
        min_makespan = self.makespan
        best_agent = None

        for a in self.model.agents:
            if hasattr(a, 'makespan') and a.makespan < min_makespan:
                min_makespan = a.makespan
                best_agent = a

        if best_agent is not None:
            self.makespan = best_agent.makespan
            self.best_schedule = {k: lst[:] for k, lst in best_agent.best_schedule.items()}
            self.current_schedule = {k: lst[:] for k, lst in best_agent.current_schedule.items()}

    def step(self):
        self.tabu_search_step()
        if self.collaboratif:
            self.contact()



class OptimisationTabuModel(Model):
    """Modèle multi-agent pour l'optimisation avec recherche tabu."""

    def __init__(self, competence_matrix, NV1, tenure=7, candidate_size=40):
        super().__init__()
        self.competence_matrix = competence_matrix
        self.C_array = normalize_competence_matrix(competence_matrix)
        self.NV1 = NV1

        # Create agents
        for i in range(NV1-1):
            a = TabuAgent(self, collaboratif=True, tenure=tenure, candidate_size=candidate_size)

        TabuAgent(self, collaboratif=False, tenure=tenure, candidate_size=candidate_size)

        self.datacollector = DataCollector(
            agent_reporters={"Makespan": lambda a: a.makespan}
        )

    def step(self):
        self.datacollector.collect(self)
        self.agents.do("step")
        self.agents.do("advance")



