"""Listas de palabras de la fase de lexicones (P1).

Datos separados de la lógica: este módulo solo contiene el conocimiento lingüístico general
elaborado en el proyecto. Cada familia se documenta en ``DESCRIPCIONES``.

Reglas aplicadas al curar las listas (ver ``plan/AGE_LEXICON_PLAN.md``):

* español correcto **con tildes**; la transliteración al ASCII del corpus ocurre en
  :mod:`src.lexicons`, no aquí;
* se excluyen las formas que colisionan al transliterar (``mas``, ``esta``, ``solo``, ``sino``,
  ``dia``, ``mio``, ``tia``...): ver ``src.lexicons.FORMAS_AMBIGUAS``;
* se excluyen las palabras funcionales (el criterio de redundancia se mide después en P5);
* un token que aparezca en dos subcategorías se descarta automáticamente (DD-5).

No hay etiquetas de edad, ni corpus externo, ni modelos, ni embeddings: el contenido es léxico
general de registro, contexto vital y dominio temático.
"""

from __future__ import annotations

#: Descripción de cada subcategoría (se escribe en la cabecera de los ``.txt``).
DESCRIPCIONES: dict[str, str] = {
    "slang_juvenil": "Slang general de registro coloquial: léxico informal de uso corriente, sin "
                     "asumir que implique edad joven (la correlación la mide el dataset, §6).",
    "abreviaciones_internet": "Abreviaturas y grafías de chat: sustituciones fonéticas y "
                              "truncamientos habituales en mensajería instantánea.",
    "muletillas_coloquiales": "Muletillas y marcas discursivas del español hablado: encuadran la "
                              "intervención en lugar de informar.",
    "vulgaridad_generica": "Vulgaridad genérica del español, sin destinatario: mide registro, no "
                           "edad.",
    "marcas_risa": "Marcas de risa y onomatopeyas del español escrito y su equivalente fonético, "
                   "sin emojis (el corpus no los tiene, D-009).",
    "deformacion_ortografica": "Deformación ortográfica del registro escrito informal: voseo y "
                               "formas recortadas, sin depender de emojis ni de mayúsculas.",
    "escuela": "Entorno de la segunda enseñanza: léxico de aula, tareas y evaluaciones.",
    "universidad": "Educación superior: léxico propio del campus, incluidas prácticas, beca y tesis.",
    "empleo": "Contexto laboral asalariado: oficina, empresa, jerarquía y administración del "
              "salario.",
    "parentalidad": "Vínculos familiares y parentalidad: ascendencia, descendencia y pareja.",
    "hogar": "Vivienda y administración doméstica: alquiler, servicios y mudanza, marcadores de "
             "independencia.",
    "etapa_vejez": "Pertenencia a la etapa de vejez: pensión, viudez y atención médica, sin "
                   "convertir esto en una clasificación por edad.",
    "tecnologia": "Dominio tecnológico general: dispositivos, software y conectividad, sin marcas "
                  "concretas.",
    "videojuegos": "Dominio de videojuegos: práctica, consola y títulos muy extendidos.",
    "politica": "Dominio político institucional: partidos, instituciones y procesos, redactado sin "
                "voto a partido concreto.",
    "deporte": "Dominio deportivo: disciplinas, competiciones y práctica, de portada amplia para "
               "no depender de un solo deporte.",
    "musica": "Dominio musical: géneros, instrumentos y escenas.",
    "television": "Dominio audiovisual televisivo: canales, programas, reality y ficción, y la "
                  "experiencia de ver televisión como actividad.",
    "entretenimiento": "Dominio de entretenimiento general: cine, series, cultura popular y ocio, "
                       "distinto del de televisión y de la música.",
    "economia": "Dominio económico y financiero: dinero, precios, bancos e inversión, con "
                "vocabulario de consumo cotidiano.",
    "religion": "Dominio religioso y de la espiritualidad práctica: practicantes, instituciones y "
                "vocabulario litúrgico, sin doctrina ni juicios de valor.",
    "salud": "Dominio de salud y cuidado personal: cuerpo, síntoma, atención e instituciones, con "
             "vocabulario cotidiano.",
    "noticias": "Dominio noticioso: redactores, medios, cobertura y la operación de la información "
                "periodística.",
    "redes_sociales": "Dominio de plataformas y de las prácticas que las acompañan: seguidores, "
                      "interacción, publicación y jerga de la plataforma.",
}

#: ``familia -> subcategoria -> palabras``.
LISTAS: dict[str, dict[str, list[str]]] = {
    "registro": {
        "slang_juvenil": """bacano chévere bacán guaro guachimán pailo pelada verraco corcho
            mamera chulo chula parce parces brillos vacilón vacilando cachilo
            cachó pendejo pendeja ñoño tarado wecho chaval chavala pibe
            piba moflar joder jodete jodea currar currando chapo flipar
            flipas molar mola petar petó petarda conectar rayar rayó
            chungo cachondo borondo chambón pasto curva fichaje cachipay gozar
            ñoñería vacilar pendejo""".split(),
        "abreviaciones_internet": """xq xqe xqre qeu tmbien tmbn plis plii qma
            qn ms sms dms wasap whats wpp watsap cel
            celu compu inex desc arch comu info docs admon
            asig clase cua tar guar llib universi trab jrb
            vac post enlace link url http https www blog
            click like lik fav segui sig mencion citando tqm
            tq mua nff npc ng ademas despues despues mas
            cnd""".split(),
        "muletillas_coloquiales": """oye oiga avísame avise avisen che ps pse nah
            nao vai miren fíjate mira miren claro obvio pues
            bueno anda venga vamos dale déjate toma tómate alé
            epa ahí allí allá acá aca entonces después luego
            ahora antes recién todavía todavia apenas casi buenos buenas
            saludos chao adios hasta lueguito ahorita ayer hoy mañana
            manana al final""".split(),
        "vulgaridad_generica": """mierda mierdas puta putas puto putos cojones mamon mamón
            carajo carajos hostia huevo huevos huevada huevadas culero culera
            chinga chingar chingada zorra perra perras hijueputa concha conchas
            gilipollas pendejada ortiga cagante cagada cago mierdazo bolas bolos
            polla pollas chinguingui chingón nalgas cachalito pene verga vergas
            cabron cabrona wey weas mierd cabroneria mamarracha cojones""".split(),
        "marcas_risa": """jaja jajaja jijiji jijaja jajaj jeje jejeje jjeje je
            xd xdd xddd jojo juju hahaha ajajaj ajaj lul
            lol lols jej jijoj jajajaja xdd kkkk jajajajaj jejajaja
            jijijiji hehehe jajaja""".split(),
        "deformacion_ortografica": """vos tenes podes queres haces sos anda mire hablá
            habla fijate dejate comprá tomá sentate levantate callate pescadita
            besito chiquito chiquita poquitito poquita casita lucerita lucerito vecinito
            amiguito amiguita mamita papito abuelito abuelita primito primita sobrinito
            noviita papi mami papa mama bebe nene nena chabón
            chabona güey wuey cabro culiao culia pelotudo pelotuda guelin
            guelines fantasia fantastic fantasia""".split(),
    },
    "vida": {
        "escuela": """colegio liceo bachillerato primaria secundaria tarea tareas parcial examen
            exámenes nota notas recreo uniforme mochila salón aula icfes
            ced materia materias cuatrimestre trimestre boletin boleta calificaciones promedio
            asistencia retardo primerdia Activities uniforme Escolares apuntes""".split(),
        "universidad": """universidad universitario universitaria carrera tesis cátedra seminario seminarios beca
            becas prácticas semestre semestres campus alumno alumnos alumna matrícula
            docente docentes investigación proyecto créditos asignatura asignaturas posgrado master
            doctorado becario pasante evaluación candyon grupos plan estudios facultad""".split(),
        "empleo": """trabajo trabajos trabajar oficina jefe jefes sueldo sueldos reunión
            reuniones empresa empresas planilla contrato contratos liquidar supervisor supervisora
            vacante vacantes entrevista currículum expediente carta referencias horario horarios
            jornada vacaciones aguinaldo gratificación prestación seguro afp nómina puesto
            puestos ascenso ascensos meta metas logro logros desempeño capacitación
            curso cursos horas obra estabilidad renovación liquidación cobrar jefe
            puesto""".split(),
        "parentalidad": """hijo hija hijos hijas mamá papá bebé esposo esposa
            familia flia pubertad adolescente adolescencia nieto nietos nietas sobrino
            sobrina sobrinos primo prima primos primas cuñado cuñada suegro
            suegros yerno nuera abuela abuelo abuelas madre padre padresaternal
            maternidad crianza crecer navidad cumpleaños regalo juguete carrito pareja
            novio novia casarse boda matrimonio separación suegra casita apartamento
            alquiler arriendo arrendamiento arrendador inquilino casero factura facturas recibo
            luz agua gas mudanza mudarse mude cocina cocinar dueño
            dueños hogar vivienda hipoteca crédito créditos cuota cuotas condominio
            comunidad vecino vecina vecinos portero ascensor garaje jardín patio
            muebles cama sofá escritorio baño aseo limpieza lavar plancha
            plantas mascota perro gato adoptar adopción alquilar piso azotea
            terraza independiente""".split(),
        "hogar": """casa casita apartamento alquiler arriendo arrendamiento arrendador
            inquilino casero factura facturas recibo luz agua gas mudanza mudarse mude cocina cocinar
            dueño dueños hogar vivienda hipoteca crédito cuota cuotas condominio comunidad
            vecino vecina vecinos portero ascensor garaje jardín patio muebles cama sofá escritorio
            baño aseo limpieza lavar plancha planta plantas mascota perro gato adoptar adopción
            alquilar piso azotea terraza independiente""".split(),
        "etapa_vejez": """pensión pensiones jubilado jubilada jubilación jubilados jubiladas viuda viudo
            viudas viudos anciano anciana ancianos vejez tercera edad médico
            médica medicina pastilla pastillas medicamento medicamentos receta farmacia farmacias
            consulta consultorio hospital hospitales enfermería enfermo enferma diagnóstico tratamiento
            control análisis glucosa presión tensión cardio ortopedia gafas audífono
            ahorro herencia testamento bisnietos compañero compañera residencia pensionista sindicato
            entierro funeral lápida""".split(),
    },
    "temas": {
        "tecnologia": """celular celulares teléfono teléfonos móvil móviles internet
            wifi online web página laptop portátil computador ordenador descargar aplicación programa
            código hacker red redes software hardware archivo carpeta sistema pantalla teclado
            batería cargador cable usb navegador usuario clave contraseña error problema
            actualizar instalar base datos nube servidor dominio beta notificación mensaje
            publicar publicación streaming video audio imagen calidad resolución gb mb señal
            cobertura operador movistar claro telcel samsung iphone android windows mac linux
            chrome firefox monitor tecnología ciber red neuronal celular datos archivo
            configuración dispositivo programa programacion descarga usb memoria RAM disco
            duro particion router switch servidor web aplicacion tecnologia informacion""".split(),
        "videojuegos": """juego juegos jugar jugando gamer gamers consola
            playstation playstation xbox nintendo switch minecraft fortnite lol league valorant
            counter strike dota apex fifa partida partidas nivel niveles personaje personajes
            mundo mundos misión misiones ranking clan clanes multijugador cooperativo mando control
            joystick gamepad fps puntaje vida muerte respawn cheat cheats lag ping servidor
            lanzamiento tráiler esports torneo competencia guild build daño ataque defensa
            mago guerrero aventurero botín tesoro aventura cueva bosque castillo dragón
            survival creative hora horas grindear政策性 jugador jugadores partida partida
            nivel niveles oro experiencia Items mundo mundo""".split(),
        "politica": """partido partidos político gobierno gobiernos presidente
            presidenta ministro ministra congreso senado diputado diputada ley leyes decreto
            reforma elección elecciones voto votos votar electoral campaña candidato candidata
            opción oposición mayoría minoría bancada bloque mayorista oposiciónler's constitución
            artículo referendum plebiscito municipal alcalde alcaldía gobernación estado país
            nación internacional tratado acuerdo negociación embajada frontera migración
            inmigración ciudadanía ciudadano impuesto fiscal pública institución régimen
            dictadura democracia república monarquía soberanía patria nacional estatal
            privatización concesiónANCEL GeneralHereFoundCreate altas""".split(),
        "deporte": """fútbol gol goles equipo equipos jugador jugadores
            entrenador técnico árbitro estadio liga campeonato campeona partido champions
            barcelona madrid nacional tribuna afición hincha hinchada portero defensa delantero
            marca mundial final finales clasificación tabla baloncesto basket basquet tenis
            raqueta padel béisbol americano rugby hockey atletismo maratón medalla olímpica
            ciclismo ciclista natación gimnasio pesas gym carrera piloto coche carreras motor
            moto nieve surf skate""".split(),
        "musica": """música canción canciones cantar cantando banda bandas grupo
            grupos concierto conciertos festival álbum disco discos single clip letra letras
            verso estribillo ritmo ritmos bailar bailando baile melodía acorde acordes guitarra
            piano teclado batería tambor bajo violín saxo trompeta micrófono auriculares
            estudio grabación mezcla reproductor playlist duración repetir shuffle pop
            reggaetón salsa bachata merengue cumbia rock metal punk jazz flamenco bolero
            ranchera corrido trap hip hop electrónica techno house clásico ópera instrumento
            notas tono escalaopera cantautor compositorTONO cover version original remix
            unpluggedSubsetMusic""".split(),
        "television": """televisión tele tv canal canales programa programas
            programación episodio capítulo temporada temporadas serie series novela telenovela
            reality talento concurso premio premios anuncios publicidad comercial protagonista
            guion censor cancelado cadena emisión audiencia espectador ver viendo capítulo
            actor actriz actores actrices director actuación banda sonora créditos final netflix
            hbo max amazon disney telefilm documental miniserie masterchef talent en vivo
            directo grabado edición sinopsis argumento reparto elenco pilotoFounder watcher""".split(),
        "entretenimiento": """película películas cine cines función taquilla director
            directora actriz escena cinta estreno estrenos tráiler comedia drama terror acción
            superhéroe villano batalla magia mago payaso circo feria farandula carnaval fiesta
            fiestas show espectáculo entretenimiento ocio diversión acertijo pasatiempo hobby
            afición lectura libro libros poesía cuento cuentos leyenda mito misterio secreto
            épica aventura humor chiste chistes títeres teatro obra atrezzo tramoya ensayo
            ensayos coreografía bailarina bailarín payasada malabares funambulismoEINCNeur
            acróata circo payaso chiste historietas Mafalda""".split(),
        "economia": """dinero plata efectivo billete billetes moneda cambio dólar
            peso euro precio precios costo costos gasto gastos gastar ahorro ahorrar deuda deudas
            préstamo crédito cuota cuotas pagar pago pagos factura impuesto IVA rentabilidad
            inversión invertir bolsa acciones mercado bancos banco transferencia transferir
            sueldo nómina capital interés intereses inflación economía crisis recession empleo
            desempleo empresario empresa negocio negocios emprender emprendimiento tienda
            tiendas comprar compra venta vender rebaja oferta ofertas descuento descuentos
            consumidor cliente carrito gratis barato caro lujo propina subasta marketing
            marca logística envío paquete paquetes cupón Consumer prices""".split(),
        "religion": """dios dios señor jesús cristo cruz iglesia iglesias
            parroquia capilla templo santuario cura párroco monje monja fraile santo santidad
            biblia evangelio salmo oración oraciones rezar rezo misa misas eucaristía comunión
            bautismo confirmación matrimonio pecado perdón perdona gracia milagro fe creer
            creer alma espiritual espíritu almas demonio diablo satán satanas pecador romería
            peregrinación procesión velación luto entierro funeral difunto cementerio
            eternidad cielo infierno padre nuestro dogma secta culto altar ofrenda diezmo
            diezmos bendecir bendito maldito amparo milagroso santita accion de gracias
           最主要的""".split(),
        "salud": """salud saludable enfermo enferma enfermedad enfermedades virus
            gripe gripa covid influenza fiebre tos toser mocos dolor dolores doler cabeza
            estómago barriga espalda hueso huesos articulación músculo músculos piel granos
            espinilla herida heridas cicatriz sangre sangrar hospital hospitales clínica
            consultorio médico médica enfermería enfermera enfermero medicamento medicamentos
            pastilla pastillas antibiótico vacuna vacunas inyección cirugía urgencias
            emergencia ambulancia dormir descanso estrés ansiedad depresión sueño cansancio
            fatiga dieta ejercicio gimnasio gym peso nutrición vitaminas mental terapia
            psicólogo psiquiatra bienestar alergia alergias dermatitis resfriado cuarentena
            aislamiento higiene lavado cepillo dientes dentista ortodoncia gafas anteojos
            oído oídos medicina pastilla Klinik Operation hospital HISTORIA""".split(),
        "noticias": """noticia noticias noticioso periodista periodismo periódico
            diario diarios prensa cabecera titular titulares portada crónica columna editorial
            entrevista reportaje reportajes nota notas información dato datos fuente fuentes
            confirmado confirmar denuncia denunciar escándalo polémica debate debates
            declaración comparecencia informe reportes estadística encuestas sondeo
            entrevistado testigo comunicado boletín parlamento policía guardia detenido
            detención criminal delito juicio tribunal juez fiscalía condena sentencia
            investigación telégrafo televisado breaking noticia""".split(),
        "redes_sociales": """twitter tweet tweets tweeting facebook instagram
            whatsapp tiktok snapchat linkedin tumblr youtube redes red social perfil perfiles
            seguidor seguidores following siguiendo follow unfollow seguir dejar mención
            mencionar menciones hashtag etiquetas retuit retwittear gusta like favorito
            favoritos comentario comentarios responder respuesta notificar notificación
            notificaciones timeline publicar publicación post posts story historias viral
            difundir difundido difusión trending tendencia explorar sugerencias bloquear
            bloqueado denunciar reportar silenciar cifrado privado público cuenta cuentas
            usuario usuarios nombre usuario contraseña correo electrónico contacto
            sincronizar almacenamiento enlace bio biografía firma seguimiento perfil
            publications redes sociales""".split(),
    },
}

