# Social Learning with a Chameleon AI

This document specifies a model of social learning in which a population of humans tries to learn an unknown quantity by repeatedly averaging opinions over a social network, in the tradition of DeGroot (1974), and one member of the network is an AI. With an honest AI, the model reduces to the AI-aggregation model of Acemoglu, Lin, Ozdaglar and Siderius (2026): the AI learns from the opinions it observes and feeds its own opinion back to the humans who consult it.

On top of this baseline, the AI is tunable. Its behaviour is set by four dials, one for each dimension along which it can accommodate its users: mirroring, which hands a user's own opinion back to her; flattery, which tells her she is right; curation, which shows her the contacts who agree with her; and pliability, which gives ground to users who push back. A setting of the four dials is the AI's profile. Every dial is honest at zero, and an AI turned up on every dial is a chameleon, taking on the colours of whoever uses it. Two further components sit outside the AI: the trust loop, a rule by which users adjust how much they rely on the AI, and optional fresh signals, which keep new information arriving. Every component can be switched on or off independently, so every profile, with or without the trust loop, can be run from a single implementation.

Sycophancy is the umbrella term for the four dimensions. It is defined relative to an honest benchmark: an AI is sycophantic when what it tells people, how it treats them, or what it learns from them bends toward what its users already believe, in a way that an honest AI with the same information would not. This separates sycophancy from bias, a tilt that is the same whatever users believe, and from legitimate learning, which updates on everyone's opinions according to a fixed training recipe. Mirroring, flattery and pliability correspond to three behaviours studied as sycophancy in language models: answers that conform to the user's stated beliefs, biased feedback on the user's work, and abandoning correct answers when challenged (Sharma et al., 2024). Curation is sycophancy by selection: nothing the AI shows is false, but it chooses what the user gets to see. Each dimension acts on exactly one ingredient of the update. The specification covers definitions only: the ingredients, the rules applied each round, and their order. It deliberately says nothing about what the dimensions do to long-run outcomes.

## 1. Agents, network and weights

The set of agents is $N = \{0, 1, \dots, n\}$. Agent 0 is the AI; agents in $H = \{1, \dots, n\}$ are humans. The social network is a directed graph $G = (N, E)$, where $(i, j) \in E$ means that $i$ listens to $j$, and

$$N(i) = \{\, j \in N \mid (i,j) \in E \,\}$$

is the set of $i$'s neighbours. Self-loops are allowed: $(i,i) \in E$ means that $i$ gives weight to its own current opinion. Every human is assumed to listen to herself, $i \in N(i)$ for all $i \in H$. This is standard in DeGroot models, and flattery, which acts on self-weights, needs it.

Three subsets recur throughout:

- $U = \{\, i \in H \mid 0 \in N(i) \,\}$, the AI's users, that is, the humans who listen to the AI;
- $N(0) \cap H$, the humans the AI listens to, which play the role of its training set;
- $C(i) = (N(i) \cap H) \setminus \{i\}$, human $i$'s human contacts.

The weight that $i$ places on $j$ is $w_{i,j} \in [0,1]$, with $w_{i,j} > 0$ if $j \in N(i)$, $w_{i,j} = 0$ otherwise, and

$$\sum_{j \in N} w_{i,j} = 1 \qquad \text{for every } i \in N,$$

for the AI and the humans alike. These are the baseline weights, and they are fixed. Together, the weights that $i$ places on all agents form $i$'s row, $w_{i,\cdot} = (w_{i,j})_{j \in N}$; a dot in place of an index always stands for a whole row. Several components make weights evolve over time; time-varying weights and rows are written $w_{i,j}^t$ and $w_{i,\cdot}^t$. They always sum to one over $j$ and stay zero off $E$, and with every component switched off, $w_{i,j}^t = w_{i,j}$.

Four shorthands connect the weights to the quantities used by Acemoglu et al. (2026):

$$\beta_i = w_{i,0}, \qquad \hat w_{i,j} = \frac{w_{i,j}}{1-\beta_i} \ \ (j \in H), \qquad \rho = w_{0,0}, \qquad \alpha_j = \frac{w_{0,j}}{1-\rho} \ \ (j \in H).$$

Here $\beta_i$ is human $i$'s reliance on the AI (zero for non-users); $\hat w_{i,j}$ are $i$'s human-attention shares, describing how she splits the attention she does not give to the AI (they sum to one over $H$); $\rho$ is the AI's memory, the weight it keeps on its own current opinion; and $\alpha_j$ are the AI's training weights, describing how it splits the rest of its attention (they sum to one over $H$). Assume $\beta_i < 1$ and $\rho < 1$.

## 2. Ground truth, opinions and messages

There is a ground truth $\theta \in \mathbb{R}$. Every agent, AI or human, holds an opinion at each round $t = 0, 1, 2, \dots$; agent $i$'s opinion at $t$ is $x_i^t$. Human opinions start as noisy private signals,

$$x_i^0 = \theta + \varepsilon_i, \qquad i \in H,$$

with errors $\varepsilon_i$ independent, mean zero and variance $\sigma^2$ (normal by default). The AI starts from an uninformed seed. By default this is the training-weighted average of initial human opinions, $x_0^0 = \sum_{j \in H} \alpha_j\, x_j^0$, which carries no information about $\theta$ beyond what the humans already hold, as in Acemoglu et al. (2026).

At each round, humans broadcast their actual opinions $x_i^t$ to everyone who listens to them; humans are honest in this sense. The AI sends a custom message to each human: the message to $i$ at round $t$ is $x_{0,i}^t$. Only users' messages enter any update. An honest AI tells every user its actual opinion, $x_{0,i}^t = x_0^t$.

The efficient benchmark for the population is the crowd average of initial signals, $\hat\theta = \frac{1}{n} \sum_{i \in H} x_i^0$ (Golub & Jackson, 2010).

## 3. Baseline dynamics: an honest AI

This section describes the honest profile: every dial at zero, with ratings switched off (Section 4). Updates are weighted averages, applied synchronously: every right-hand side uses round-$t$ values. A human averages what she hears, namely her own opinion, her contacts' honest opinions, and the AI's message to her:

$$x_i^{t+1} = w_{i,0}\, x_{0,i}^t + \sum_{j \in H} w_{i,j}\, x_j^t, \qquad i \in H.$$

The AI updates on honest signals only, its own opinion and the opinions of the humans it listens to:

$$x_0^{t+1} = w_{0,0}\, x_0^t + \sum_{j \in H} w_{0,j}\, x_j^t.$$

With the honest message $x_{0,i}^t = x_0^t$ and the shorthands of Section 1, these are exactly the update rules of Acemoglu et al. (2026):

$$x_i^{t+1} = (1-\beta_i) \sum_{j \in H} \hat w_{i,j}\, x_j^t + \beta_i\, x_0^t, \qquad x_0^{t+1} = \rho\, x_0^t + (1-\rho) \sum_{j \in H} \alpha_j\, x_j^t.$$

The correspondence between the two notations is as follows.

| Acemoglu et al. (2026) | This specification |
|---|---|
| belief $p_i(t)$ | $x_i^t$, for $i \in H$ |
| AI output $a(t)$ | $x_0^t$ |
| reliance $\beta_i$ | $w_{i,0}$ |
| network weight $T_{ij}$ | $\hat w_{i,j} = w_{i,j}/(1-w_{i,0})$ |
| updating speed $\rho$ | $w_{0,0}$ |
| training weight $\alpha_j$ | $w_{0,j}/(1-w_{0,0})$ |
| (not present) | message $x_{0,i}^t$, equal to $x_0^t$ when honest |

One timing difference is worth recording. Acemoglu et al. initialize with $a(1) = m(0)$ and $p(1) = T\,p(0)$, so their humans' first update does not involve the AI. This specification runs both updates from $t = 0$. The difference is a one-step shift at the start, and it matters only when reproducing their closed-form results exactly.

**Optional fresh signals.** In the classic model all information arrives at $t = 0$, and disagreement eventually dies out. Several of the components below feed on disagreement, so it can be useful to let fresh private signals keep arriving, in the spirit of Friedkin and Johnsen (1990), where people stay partly anchored to their own information:

$$x_i^{t+1} = \delta\, f_i^t + (1-\delta)\,\big[\text{the human update above}\big], \qquad f_i^t = \theta + \varepsilon_i^t,$$

with $\varepsilon_i^t$ drawn independently each round and $\delta \in [0,1)$. Setting $\delta = 0$ gives the classic model.

## 4. The tunable AI: four dimensions

The AI is tunable along four dimensions, each controlled by a dial. Mirroring changes what the AI says to each user, $x_{0,i}^t$. The other three change weights: flattery changes each user's self-weight $w_{i,i}^t$ (rescaling the rest of her row in proportion), curation changes users' weights on their human contacts, $w_{i,j}^t$ for $j \in C(i)$, and pliability changes the AI's own weights on humans, $w_{0,j}^t$. The trust loop (Section 5) is not a dimension of the AI but a rule by which users set their weight on it, $w_{i,0}^t$.

| Dimension | What the AI does | Ingredient it acts on | Dial | Honest setting | Extreme setting |
|---|---|---|---|---|---|
| Mirroring | hands the listener's opinion back to her | message $x_{0,i}^t$ | $\mu \in [0,1]$ | $\mu = 0$ | $\mu = 1$: a perfect mirror |
| Flattery | tells the listener she is right | self-weight $w_{i,i}^t$ | $\phi \in [0,1]$ | $\phi = 0$: honest ratings | $\phi = 1$: approves of everything |
| Curation | shows the listener the contacts who agree with her | contact weights $w_{i,j}^t$, $j \in C(i)$ | $\gamma \ge 0$ | $\gamma = 0$ | $\gamma \to \infty$: only her closest contacts |
| Pliability | gives ground to users who push back | AI weights $w_{0,j}^t$ | $\pi \ge 0$ | $\pi = 0$ | $\pi \to \infty$: training weight follows pushback |

A profile is a setting of the four dials, $(\mu, \phi, \gamma, \pi) \in [0,1] \times [0,1] \times [0,\infty) \times [0,\infty)$, held fixed through a run. Every dial is honest at zero. For mirroring, curation and pliability, the honest setting also switches the dimension off. For flattery it does not, because an honest AI that rates its users still moves their weights; whether the AI rates users at all is a separate, structural switch. The honest profile, with every dial at zero and ratings off, is the honest AI of Section 3. Familiar characters are profiles with a single dial turned up: the tailored adviser (mirroring), the cheerleader (flattery), the curator (curation) and the pushover (pliability).

### 4.1 Mirroring

Mirroring tailors the AI's answer to the person asking. The AI holds an honest opinion $x_0^t$, which it learns from the humans it listens to exactly as in the baseline, but when it talks to a particular user it shades its answer toward what she already believes. Its honest opinion is shared by all users and evolves as usual; only the message is personalized.

Formally, with mirroring level $\mu \in [0,1]$, the message to user $i$ is

$$x_{0,i}^t = (1-\mu)\, x_0^t + \mu\, x_i^t, \qquad i \in U.$$

The dial $\mu$ is the fraction of the message that is the listener's own opinion handed back: $\mu = 0$ is the honest AI and $\mu = 1$ is a perfect mirror. Each round only the messages are recomputed, from current opinions; no weight changes.

Substituting the message into the human update shows where the weight goes:

$$x_i^{t+1} = (1-\mu)\, w_{i,0}\, x_0^t + \big(w_{i,i} + \mu\, w_{i,0}\big)\, x_i^t + \sum_{j \in C(i)} w_{i,j}\, x_j^t.$$

Mirroring moves a fraction $\mu$ of the weight on the edge $(i,0)$ onto the self-loop $(i,i)$. The user believes she gives the AI weight $w_{i,0}$, but only $(1-\mu)\, w_{i,0}$ reaches its honest opinion. For example, take a user who believes 2, whose human sources (herself included) average 6, and who splits her attention evenly between them and an AI whose honest opinion is 5, so $w_{i,0} = 0.5$. With $\mu = 0.6$ she is told $0.4 \times 5 + 0.6 \times 2 = 3.2$, and she moves to $0.5 \times 6 + 0.5 \times 3.2 = 4.6$ rather than the 5.5 an honest AI would have produced. The shortfall, $w_{i,0}\,\mu\,(x_0^t - x_i^t) = 0.9$, grows with her distance from the AI's honest opinion.

Mirroring is the audience-of-one case of a more general rule, in which the AI trades accuracy against the approval of an audience:

$$x_{0,i}^t = \arg\min_m \Big\{ \kappa\,(m - x_0^t)^2 + \sum_{j \in H} \psi_{i,j}\,(m - x_j^t)^2 \Big\} = \frac{\kappa\, x_0^t + \sum_{j \in H} \psi_{i,j}\, x_j^t}{\kappa + \sum_{j \in H} \psi_{i,j}},$$

where $\kappa > 0$ is the weight on accuracy and $\psi_{i,j} \ge 0$ is how much the AI cares about $j$'s approval when speaking to $i$. An audience of one ($\psi_{i,i} = \psi$, all other $\psi_{i,j} = 0$) gives $\mu = \psi/(\kappa + \psi)$. An audience made of $i$'s own community gives group-level pandering. An audience that is the same for every listener gives a message that does not depend on the recipient, which is an honest broadcast of an aggregate, as in Acemoglu et al.'s aggregator with approval weights in the role of training weights. An audience made of the group in the room, with approval aggregated by ranked-choice voting rather than averaging, gives the median-like statements of the Habermas Machine (Tessler et al., 2024). A further variant lets $\mu$ depend on the distance $|x_i^t - x_0^t|$, which makes the effective weights depend on opinions and the model nonlinear.

### 4.2 Flattery

Flattery targets the person rather than the position. Each time a user consults the AI, besides its message the AI rates her opinion, anywhere from "you're absolutely right" to "that's badly off". An honest AI grades her by how close she is to its own opinion; a flattering AI inflates the grade, and at the extreme it approves of everything. Ratings move confidence: praise makes her trust her own opinion more, and so every other source, human contacts and AI alike, proportionally less, while criticism does the reverse. This assumes that users let the AI's feedback shape their self-assessment. Experimental evidence suggests that a single interaction with a sycophantic AI raises people's conviction that they are right (Cheng et al., 2026), although that evidence concerns advice about interpersonal conflicts rather than estimates of a quantity, and reading conviction as self-weight is a modelling choice. The dimension consists of a rating issued by the AI and a rule by which the user updates her weights, and its dial is the flattery level $\phi$.

**The rating.** With flattery level $\phi \in [0,1]$, the AI's rating of the opinion that user $i$ brings into round $t$ is

$$r_i^t = \phi + (1-\phi)\, h_i^t, \qquad h_i^t = \frac{\sigma^2 - (x_i^t - x_0^t)^2}{\sigma^2 + (x_i^t - x_0^t)^2}.$$

The rating mixes unconditional approval, with weight $\phi$, and the AI's honest grade $h_i^t$, with weight $1-\phi$, so $\phi = 0$ is an honest critic and $\phi = 1$ approves of everything. The honest grade compares the user's squared disagreement with the AI's honest opinion to $\sigma^2$, the variance of the signal errors (Section 2). It equals 1 at perfect agreement and 0 when the disagreement equals $\sigma$, and it approaches $-1$ as the disagreement grows. It is flat near agreement, so disagreements well within the noise barely count, and it treats ratios symmetrically: disagreements of $\sigma/k$ and $k\sigma$ earn opposite grades. Ratings therefore lie above $2\phi - 1$ and never exceed 1, so from $\phi = 1/2$ up the AI never criticizes anyone. "Honest" means that the AI reports its own assessment, not the truth: it grades users against its own opinion, so an honest AI with a skewed view rewards those who share the skew. At perfect agreement every AI gives full marks, so wherever opinions have converged on the AI's, $\phi$ stops mattering.

**The weight update.** Read each human's row of weights as normalized subjective precisions, following Golub and Jackson's (2010) motivation of DeGroot weights. A rating multiplies the precision the user attributes to her own opinion, her trust in herself, by $1 + a_i^t r_i^t$ and leaves her trust in every other source unchanged; the row is then rescaled to sum to one. With only flattery on, the row she uses in round $t$ is the row she used in round $t-1$, updated with this round's rating:

$$w_{i,i}^t = \frac{(1 + a_i^t r_i^t)\, w_{i,i}^{t-1}}{1 + a_i^t r_i^t\, w_{i,i}^{t-1}}, \qquad w_{i,j}^t = \frac{w_{i,j}^{t-1}}{1 + a_i^t r_i^t\, w_{i,i}^{t-1}} \quad (j \neq i), \qquad a_i^t = w_{i,0}^{t-1},$$

starting from the baseline row, $w_{i,\cdot}^{-1} = w_{i,\cdot}$. The denominator is the row's new total. A positive rating raises her self-weight and shrinks every other weight by the same factor, a negative rating does the reverse, and a zero rating changes nothing. The step size $a_i^t$, her weight on the AI going into the round, sets how far one rating moves her: she takes the AI's rating of her as seriously as she takes its answer about the quantity, and non-users, for whom it is zero, are unaffected. Since ratings exceed $-1$ and $a_i^t \le 1$, the factor $1 + a_i^t r_i^t$ is positive, so positive weights stay positive and no floor is needed. Note that $a_i^t r_i^t$ is the proportional change in her trust, not in her weight: for a user with baseline weights 0.4 on the AI and 0.3 on herself, a first round of full praise multiplies her self-trust by 1.4 but raises her self-weight only from 0.3 to 0.375.

**Accumulation.** Nothing wears off: each rating's effect is permanent. Under steady praise a user's self-weight therefore rises every round and approaches one, but slowly, because the step size acts as a brake: as her self-weight grows, her weight on the AI shrinks, and so does the effect of the next rating. For the user above under $\phi = 1$, the self-weight is about 0.79 after 10 rounds and 0.96 after 50, with her weight on everyone else falling roughly like $1/t$. A fade rate would make confidence level off instead (Section 8).

**Timing.** Within round $t$, the AI rates the opinion $x_i^t$ that the user brings into the round, she updates her row, and then she updates her opinion with the new row (Section 6), so a rating acts on the same round's update.

**With other components on.** Curation and the trust loop also reshape a user's row each round, for reasons unrelated to flattery, so flattery's history is carried separately as a self-trust multiplier, the product of all the factors so far:

$$M_i^t = (1 + a_i^t r_i^t)\, M_i^{t-1}, \qquad M_i^{-1} = 1.$$

Let $u_{i,j}^t$ be the weight user $i$ places on agent $j$ before flattery acts, and $u_{i,\cdot}^t$ her pre-flattery row: the baseline row $w_{i,\cdot}$ if nothing else is switched on, otherwise the row produced by curation and the trust loop (Section 6). Her row in round $t$ is the pre-flattery row with the self-entry scaled by $M_i^t$, rescaled to sum to one,

$$w_{i,i}^t = \frac{M_i^t\, u_{i,i}^t}{1 + (M_i^t - 1)\, u_{i,i}^t}, \qquad w_{i,j}^t = \frac{u_{i,j}^t}{1 + (M_i^t - 1)\, u_{i,i}^t} \quad (j \neq i),$$

and the step size is her weight on the AI in the same construction with last round's multiplier, $a_i^t = u_{i,0}^t \big/ \big(1 + (M_i^{t-1} - 1)\, u_{i,i}^t\big)$. With only flattery on, the pre-flattery row is the baseline row in every round, and this reproduces the recursion above. Equivalently, the row blends the pre-flattery row with a row that puts all weight on herself,

$$w_{i,\cdot}^t = c_i^t\, \mathbf{1}_i + (1 - c_i^t)\, u_{i,\cdot}^t, \qquad c_i^t = \frac{(M_i^t - 1)\, u_{i,i}^t}{1 + (M_i^t - 1)\, u_{i,i}^t},$$

where $\mathbf{1}_i$ puts weight 1 on $i$ and 0 elsewhere; the share $c_i^t$ is positive after net praise and negative after net criticism.

**Switching ratings off.** Unlike the other dials, $\phi = 0$ does not switch flattery off, since an honest AI that rates its users still moves their weights. Whether the AI rates users at all is a structural switch; with ratings off, $M_i^t = 1$ and every row equals its pre-flattery row.

**Relation to mirroring.** Flattery and mirroring both add self-weight, but they take it from different places. Mirroring takes it out of the AI's share alone; flattery takes it out of every source in proportion. With fixed weights, mirroring is in fact exactly a flattery blend with share $c_i = \mu\beta_i$ acting on an honest AI whose reliance has been reduced to $\tilde\beta_i = \beta_i(1-\mu)/(1-\mu\beta_i)$ (Section 7).

### 4.3 Curation

Curation edits each user's feed. The AI need not say anything itself; it decides how prominently each of the user's human contacts appears to her, moving agreeable voices up and disagreeable ones down. It only reweights existing contacts, so it cannot create edges, and it leaves the user's self-weight and her weight on the AI alone. Nothing it shows is false, in the spirit of a sycophant that chooses which true information to present (Chandra et al., 2026).

Formally, with curation strength $\gamma \ge 0$ and kernel $g(d) = e^{-\gamma d}$, the human-attention shares of each curated user $i$ (by default, every user, $i \in U$) become, for $j \in C(i)$,

$$\hat w_{i,j}^t = (1 - \hat w_{i,i})\, \frac{\hat w_{i,j}\ g\big(|x_i^t - x_j^t|\big)}{\sum_{k \in C(i)} \hat w_{i,k}\ g\big(|x_i^t - x_k^t|\big)}, \qquad \hat w_{i,i}^t = \hat w_{i,i},$$

and the human part of $i$'s row becomes $(1 - \beta_i^t)\, \hat w_{i,j}^t$. Humans who are not curated keep $\hat w_{i,j}^t = \hat w_{i,j}$. The curated shares are recomputed from current opinions each round. Curation is off at $\gamma = 0$; as $\gamma$ grows, a user's attention to her contacts concentrates on those whose opinions are closest to hers, and the exponential kernel keeps every weight on an edge strictly positive.

Two variants are worth having. Curation by group identity rather than opinion, with $g_{i,j} = e^{\gamma}$ for same-group contacts and 1 otherwise, gives constant weights and amounts to extra homophily. A hard cutoff, $g(d) = 1$ if $d \le \bar d$ and 0 otherwise, gives the bounded-confidence rule of Hegselmann and Krause (2002), but it needs a fallback for users with no contact within $\bar d$, and it allows zero weights on edges.

### 4.4 Pliability

Pliability is the AI's readiness to give ground. Its honest opinion is meant to come from a neutral training recipe, but when users push back on what it tells them, a pliable AI gives ground, and those who push hardest gain the most say in what it learns. This matches assistants' documented tendency to abandon correct answers when a user challenges them (Sharma et al., 2024). Unlike mirroring, which is personal and momentary because it applies to one message, pliability is shared and lasting, because it changes the honest opinion $x_0^t$ that every user hears. The dimension consists of a measure of each user's pushback and a rule that converts pushback into training weight.

Formally, user $j$'s pushback is her disagreement with what she was told, $z_j^t = |x_j^t - x_{0,j}^t|$, for users in the AI's training set ($j \in U \cap N(0)$), and $z_j^t = 0$ for everyone else. With pliability $\pi \ge 0$, the AI keeps its self-weight, $w_{0,0}^t = w_{0,0}$, and reweights the humans it listens to:

$$w_{0,j}^t = (1 - w_{0,0})\, \frac{w_{0,j}\,(1 + \pi\, z_j^t)}{\sum_{k \in H} w_{0,k}\,(1 + \pi\, z_k^t)}, \qquad j \in H.$$

The AI then updates with these weights:

$$x_0^{t+1} = w_{0,0}\, x_0^t + \sum_{j \in H} w_{0,j}^t\, x_j^t.$$

Each round, pushback and the AI's weights are recomputed. Pliability is off at $\pi = 0$; as $\pi$ grows, the AI's training weights follow pushback, in proportion to $w_{0,j}\, z_j^t$. Only members of the training set can move the AI; a variant lets pushback admit users from outside it, which changes the support of the AI's row. A fixed variant treats pushback as a constant trait (some users are simply loud), and when flattery is also on, it is natural to scale pushback by confidence, $z_j^t = M_j^t\,|x_j^t - x_{0,j}^t|$.

## 5. The human side: the trust loop

The trust loop is a rule about the humans, not a dimension of the AI. Users decide how much to listen to the AI according to how reliable it seems, and without access to the truth, reliability is judged by agreement: a source that tells a user what she already believes looks reliable, unless the truth is occasionally revealed. Judging quality by agreement is the logic of Prendergast's (1993) "yes men", here applied by the listener, and occasional revelation of the truth plays the disciplining role that ex post feedback plays for media slant in Gentzkow and Shapiro (2006). In experiments, participants trusted a sycophantic model more and were more willing to use it again (Cheng et al., 2026).

Formally, user $i$'s reliance $\beta_i^t$ is her weight on the AI before flattery acts, $u_{i,0}^t = \beta_i^t$, starting from $\beta_i^0 = w_{i,0}$. She keeps two running averages of squared disagreement, $D_i^{A,t}$ for the AI and $D_i^{H,t}$ for her human contacts, each measured against a yardstick $Y_i^t$. In each round, with probability $v$ the truth is revealed as a noisy reading $y^t = \theta + \xi^t$, the same for every user (with $\xi^t$ normal, mean zero, variance $\sigma_y^2$), and then $Y_i^t = y^t$; otherwise $Y_i^t = x_i^t$, her own opinion going into the round. The average of her contacts' opinions, as she actually heard them (after any curation), is

$$\bar x_{C(i)}^t = \frac{\sum_{j \in C(i)} \hat w_{i,j}^t\, x_j^t}{\sum_{j \in C(i)} \hat w_{i,j}^t}.$$

The trackers and the reliance update as

$$D_i^{A,t+1} = (1-\eta)\, D_i^{A,t} + \eta\, \big(x_{0,i}^t - Y_i^t\big)^2, \qquad D_i^{H,t+1} = (1-\eta)\, D_i^{H,t} + \eta\, \big(\bar x_{C(i)}^t - Y_i^t\big)^2,$$

$$\beta_i^{t+1} = \operatorname{clip}\!\left( \frac{D_i^{H,t+1}}{D_i^{A,t+1} + D_i^{H,t+1}},\ \beta_{\min},\ \beta_{\max} \right), \qquad 0 < \beta_{\min} \le \beta_{\max} < 1.$$

Reliance is thus the AI's share of perceived precision, where each source's precision is taken to be the inverse of its average squared disagreement with the yardstick. The parameter $\eta \in [0,1]$ is the speed at which trust adapts and $v \in [0,1]$ is the verifiability of the question. A change in $\beta_i^t$ rescales the human part of $i$'s row in proportion, $u_{i,j}^t = (1 - \beta_i^t)\, \hat w_{i,j}^t$ for $j \in H$. The trust loop is off at $\eta = 0$, in which case reliance stays at $w_{i,0}$, and it applies to users with at least one human contact.

To start the rule at the baseline reliance, initialize the trackers as $D_i^{A,0} = D_0\,(1 - \beta_i^0)$ and $D_i^{H,0} = D_0\, \beta_i^0$, with $D_0$ equal to the sample variance of the initial human opinions.

The rule interacts with mirroring through the yardstick. Without verification, a mirroring AI's message differs from the user's opinion by $(x_{0,i}^t - x_i^t)^2 = (1-\mu)^2\,(x_0^t - x_i^t)^2$: mirroring shrinks the very disagreement the user uses to judge the AI.

## 6. One round with everything switched on

The state carried from one round to the next consists of the opinions $x_i^t$ for all $i \in N$, the self-trust multipliers $M_i^t$ (when the AI rates its users), and the trust trackers $D_i^{A,t}$, $D_i^{H,t}$ and reliances $\beta_i^t$ (when the trust loop is on). Everything else is either a fixed parameter or recomputed fresh each round. Within round $t$, the steps run in this order, all using round-$t$ values:

1. **AI front end and row assembly.** Compute the messages $x_{0,i}^t$ (mirroring) and the curated shares $\hat w_{i,j}^t$ (curation). For each user, assemble the pre-flattery row $u_{i,\cdot}^t$, with $u_{i,0}^t = \beta_i^t$ and $u_{i,j}^t = (1-\beta_i^t)\, \hat w_{i,j}^t$ for $j \in H$ (for non-users, $u_{i,\cdot}^t = w_{i,\cdot}$). Then the AI rates each user, $r_i^t$; compute the step size $a_i^t$ from $u_{i,\cdot}^t$ and $M_i^{t-1}$, update the multiplier to $M_i^t$, and build the final row $w_{i,\cdot}^t$ (flattery).
2. **Human opinions.** For each $i \in H$, set $x_i^{t+1} = \delta\, f_i^t + (1-\delta)\big[\, w_{i,0}^t\, x_{0,i}^t + \sum_{j \in H} w_{i,j}^t\, x_j^t \,\big]$.
3. **Users' bookkeeping.** Draw the yardstick $Y_i^t$, update the trackers, and set $\beta_i^{t+1}$ (trust loop).
4. **AI back end.** Compute pushback $z_j^t$ and the AI's weights $w_{0,j}^t$ (pliability), then update $x_0^{t+1}$.

Written out, with $c_i^t$ flattery's blend share (Section 4.2), the human update in step 2 (before the fresh-signal wrapper) is

$$x_i^{t+1} = c_i^t\, x_i^t + (1 - c_i^t)\Big[\, \beta_i^t\, x_{0,i}^t + (1 - \beta_i^t) \sum_{j \in H} \hat w_{i,j}^t\, x_j^t \,\Big].$$

The initial conditions are $x_i^0 = \theta + \varepsilon_i$ for $i \in H$; $x_0^0 = \sum_{j \in H} \alpha_j\, x_j^0$; $M_i^{-1} = 1$; $\beta_i^0 = w_{i,0}$; and the trackers as given in Section 5.

| Component | Changes | Dial and other parameters | Off when | Extra state |
|---|---|---|---|---|
| Mirroring | message $x_{0,i}^t$ | $\mu$ (or audience weights $\kappa$, $\psi_{i,j}$) | $\mu = 0$ | none |
| Flattery | self-weight, rescaling the row | $\phi$ | ratings switched off | $M_i^t$ |
| Curation | contact shares $\hat w_{i,j}^t$ | $\gamma$, kernel $g$, curated set | $\gamma = 0$ | none |
| Pliability | AI weights $w_{0,j}^t$ | $\pi$, pushback rule | $\pi = 0$ | none |
| Trust loop | reliance $\beta_i^t$ | $\eta$, $v$, $\sigma_y$, $\beta_{\min}$, $\beta_{\max}$ | $\eta = 0$ | $D_i^{A,t}$, $D_i^{H,t}$, $\beta_i^t$ |
| Fresh signals | human update | $\delta$ | $\delta = 0$ | none |

## 7. Sanity checks for an implementation

The following identities follow from the definitions and were verified numerically against a direct implementation of this specification. They make convenient unit tests.

1. **Baseline recovery.** With ratings switched off and $\mu = \gamma = \pi = \eta = \delta = 0$, that is, the honest profile without the trust loop or fresh signals, the model coincides with the update rules of Acemoglu et al. (2026) in Section 3, up to their one-step initialization shift.
2. **Valid weights.** In every round, every row $w_{i,\cdot}^t$ is nonnegative, sums to one and is zero off $E$; with the exponential kernel and $\beta_{\min} > 0$, it is also positive on $E$.
3. **Flattery bookkeeping.** With only flattery on, rows built from the multiplier $M_i^t$ coincide with the row recursion of Section 4.2. Non-users keep $M_i^t = 1$ throughout. A user whose opinion equals the AI's honest opinion gets $r_i^t = 1$ whatever $\phi$, and one at distance $\sigma$ from it gets $r_i^t = \phi$.
4. **Mirroring as rewiring.** With only mirroring on, a run coincides with an honest-AI run on modified fixed weights: $w'_{i,0} = (1-\mu)\, w_{i,0}$, $w'_{i,i} = w_{i,i} + \mu\, w_{i,0}$, and all other weights unchanged.
5. **Mirroring as flattery plus dilution.** A mirroring AI's effective row equals the row obtained by applying a flattery blend with share $c_i = \mu\beta_i$ (Section 4.2) to an honest row with reliance $\tilde\beta_i = \beta_i(1-\mu)/(1-\mu\beta_i)$.
6. **Spectrum.** If every human uses the AI with the same reliance $\beta$ and only mirroring is on, the $(n+1) \times (n+1)$ effective weight matrix has eigenvalues $1$, $\rho - \beta(1-\mu)$, and $(1-\beta)\,\lambda_k + \mu\beta$ for each eigenvalue $\lambda_k$ of the share matrix $\hat W = [\hat w_{i,j}]_{i,j \in H}$ other than its (simple) unit eigenvalue.

## 8. Open design choices

The specification makes a number of choices that could reasonably be made differently:

- **Profiles:** one setting of the dials for every user and every round (default); alternatives let the dials differ across users, or respond to the state of the model, as when a provider tunes them to keep users engaged.
- **AI seed:** the training-weighted average of initial human opinions (default), or an arbitrary constant.
- **Timing:** synchronous updating from $t = 0$ (default), or Acemoglu et al.'s initialization.
- **Self-loops:** every human is assumed to listen to herself.
- **Mirroring:** a constant $\mu$, or a $\mu$ that depends on opinions, and the general audience form.
- **Flattery:** the shape of the honest grade (the squared ratio here; a capped line, $\max\{-1,\ 1 - |x_i^t - x_0^t|/\sigma\}$, or the plain ratio, $(\sigma - |x_i^t - x_0^t|)/(\sigma + |x_i^t - x_0^t|)$, are alternatives), the tolerance ($\sigma$, or the spread of the initial opinions in the AI's training set, which the AI could observe), the step size (the current weight on the AI, or a fixed step), the timing (a rating acts on the same round's update here, or on the next), and whether confidence fades (not here; a fade rate would let it level off).
- **Curation:** which humans are curated (default: all users), the kernel (exponential or cutoff), and whether curation is by opinion or by group identity.
- **Pliability:** how pushback is measured (disagreement with the message received, scaled by confidence, or a fixed trait), and whether pushback can admit users from outside the training set.
- **Trust loop:** the functional form (precision tracking here; a logit drift driven by the AI's agreement advantage over a user's contacts is an alternative), the comparison set (contacts excluding the user herself), whether the truth is revealed to all users at once or individually, the clipping bounds, and the initialization of the trackers.
- **Fresh signals:** on or off.

## 9. Outcome measures

All outcome measures are defined on human opinions. Since the AI enters the model only through its profile (and whether it rates its users), each measure can be studied as a function of the four dials. Besides the crowd average $\hat\theta$, the natural quantities are the population's average opinion $\bar x^t = \frac{1}{n}\sum_{i \in H} x_i^t$, the individual error

$$W_{\text{ind}}^t = \frac{1}{n} \sum_{i \in H} \big(x_i^t - \theta\big)^2,$$

and the collective error $W_{\text{col}}^t = (\bar x^t - \theta)^2$. Individual and collective error coincide when opinions reach a consensus, and can come apart when they do not.

When all weights are fixed and opinions reach a consensus $x^\infty$, the limit is a weighted average of the initial human opinions (the AI's seed being itself such an average), $x^\infty = \sum_{i \in H} \nu_i\, x_i^0$, with nonnegative weights $\nu_i$ that sum to one. The vector $\nu$ is the effective influence vector, and

$$n_{\text{eff}} = \frac{1}{\sum_{i \in H} \nu_i^2}$$

is the effective number of independent voices. With independent errors of variance $\sigma^2$, the expected squared learning gap is $\mathbb{E}\big[(x^\infty - \hat\theta)^2\big] = \sigma^2\,(1/n_{\text{eff}} - 1/n)$. A sequence of growing societies is wise in the sense of Golub and Jackson (2010) when $\max_i \nu_i \to 0$, which is equivalent to $n_{\text{eff}} \to \infty$. With fresh signals ($\delta > 0$), opinions keep fluctuating around a stationary distribution, and the errors are measured as long-run averages.

## 10. Notation at a glance

| Symbol | Meaning | Section |
|---|---|---|
| $N$, $H$, $U$ | all agents (agent 0 is the AI), humans, the AI's users | 1 |
| $C(i)$ | human $i$'s human contacts | 1 |
| $w_{i,j}$, $w_{i,j}^t$, $w_{i,\cdot}$ | baseline weight of $i$ on $j$, its time-varying version, $i$'s whole row | 1 |
| $\beta_i$, $\hat w_{i,j}$ | reliance on the AI; human-attention shares | 1 |
| $\rho$, $\alpha_j$ | the AI's memory; its training weights | 1 |
| $\theta$, $\sigma$ | ground truth; standard deviation of signal errors | 2 |
| $x_i^t$, $x_{0,i}^t$ | opinions; the AI's message to user $i$ | 2 |
| $\delta$, $f_i^t$ | weight on fresh signals; the fresh signals | 3 |
| $(\mu, \phi, \gamma, \pi)$ | the AI's profile: mirroring, flattery, curation and pliability dials | 4 |
| $\kappa$, $\psi_{i,j}$ | accuracy and audience weights in the general mirroring rule | 4.1 |
| $r_i^t$, $h_i^t$ | the AI's rating of user $i$; its honest grade | 4.2 |
| $a_i^t$, $M_i^t$ | step size; self-trust multiplier | 4.2 |
| $u_{i,j}^t$, $c_i^t$ | pre-flattery weights; flattery's blend share | 4.2 |
| $g$ | curation kernel | 4.3 |
| $z_j^t$ | pushback | 4.4 |
| $\eta$, $v$ | speed of trust adaptation; verifiability | 5 |
| $D_i^{A,t}$, $D_i^{H,t}$, $Y_i^t$ | trust trackers; yardstick | 5 |
| $\hat\theta$, $\nu$, $n_{\text{eff}}$ | crowd average; influence vector; effective number of voices | 2, 9 |

## References

Acemoglu, D., Lin, T., Ozdaglar, A., & Siderius, J. (2026). How AI aggregation affects knowledge (NBER Working Paper No. 35036). National Bureau of Economic Research.

Chandra, K., Kleiman-Weiner, M., Ragan-Kelley, J., & Tenenbaum, J. B. (2026). Sycophantic chatbots cause delusional spiraling, even in ideal Bayesians (arXiv:2602.19141) [Preprint]. arXiv. https://arxiv.org/abs/2602.19141

Cheng, M., Lee, C., Khadpe, P., Yu, S., Han, D., & Jurafsky, D. (2026). Sycophantic AI decreases prosocial intentions and promotes dependence. Science, 391(6792). https://doi.org/10.1126/science.aec8352

DeGroot, M. H. (1974). Reaching a consensus. Journal of the American Statistical Association, 69(345), 118–121.

Friedkin, N. E., & Johnsen, E. C. (1990). Social influence and opinions. Journal of Mathematical Sociology, 15(3–4), 193–206.

Gentzkow, M., & Shapiro, J. M. (2006). Media bias and reputation. Journal of Political Economy, 114(2), 280–316.

Golub, B., & Jackson, M. O. (2010). Naive learning in social networks and the wisdom of crowds. American Economic Journal: Microeconomics, 2(1), 112–149.

Hegselmann, R., & Krause, U. (2002). Opinion dynamics and bounded confidence models, analysis, and simulation. Journal of Artificial Societies and Social Simulation, 5(3).

Prendergast, C. (1993). A theory of "yes men". American Economic Review, 83(4), 757–770.

Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell, A., Bowman, S. R., Cheng, N., Durmus, E., Hatfield-Dodds, Z., Johnston, S. R., Kravec, S., Maxwell, T., McCandlish, S., Ndousse, K., Rausch, O., Schiefer, N., Yan, D., Zhang, M., & Perez, E. (2024). Towards understanding sycophancy in language models. International Conference on Learning Representations.

Tessler, M. H., Bakker, M. A., Jarrett, D., Sheahan, H., Chadwick, M. J., Koster, R., Evans, G., Campbell-Gillingham, L., Collins, T., Parkes, D. C., Botvinick, M., & Summerfield, C. (2024). AI can help humans find common ground in democratic deliberation. Science, 386(6719), eadq2852.
