# procon

1. algorithm saijruulna(fps)
2. tatah, stickman igg huduguungu bailgah jishee
3. unah hurd tootsooloh
4. nuuriig tanij hucilturugc hemjdgee saijruulah

1. mashin surgah ai \muuguntsur bakter hugts\
2. zahialga gargah \ heregtei tehnik tuhuurumjiinhuu\
3. bakter mugntsr zergee haraad ilruuleh

Нийгмийн асуудал: Яагаад энэ төслийг хийх болсон шалтгаан (Work style reform, хөгшрөлт).

Технологийн шийдэл: Чиний өмнө нь хийж байсан YOLO, FaceMesh болон шинээр нэмэгдэх Whisper AI-ийн үүрэг.

Шүүгчдийг гайхшруулах гол цэг: "Task-Shifting" буюу ажил шилжүүлэлт хэрхэн явагдаж байгааг харуулсан хүснэгт.

Төлөвлөгөө: Багийнхан хэн, юуг, хэдий хугацаанд хийх тухай хөгжүүлэлтийн дараалал.

Дүгнэлт: Энэ төсөл яагаад Kosen Procon дээр ялах боломжтойг уриалсан санамж зэргийг бүрэн багтаасан болно.

To get a 100% free API with zero usage caps, zero subscription fees, and complete data privacy, the solution is to use an open-source model. You download the model weights for free, wrap them in a lightweight API server, and run it on your own computer or cloud server.For 70+ languages, the open-source ecosystem has two massive players that easily hit your requirement.1. OpenAI Whisper (Large-v3 / Turbo)Despite OpenAI being a commercial company, they fully open-sourced the weights for the Whisper model under the permissive MIT license.  Language Count: Supports 99+ languages out of the box.  Performance: It is the undisputed gold standard for open-source speech recognition. It effortlessly handles accents, automatically detects what language is being spoken, and can even translate those 99+ languages directly into English text.  Efficiency: Highly optimized variations like Whisper Large-v3 Turbo or Distil-Whisper give you top-tier accuracy while running incredibly fast on consumer hardware or budget-friendly GPUs.  2. Meta MMS (Massive Multilingual Speech)If Whisper doesn't cover a highly specific regional dialect you need, Meta's open-source MMS model is the ultimate backup.Language Count: Supports over 1,100 languages.  Performance: While Whisper is generally preferred for standard conversational accuracy in major world languages, Meta MMS is entirely open, free, and completely unmatched in its linguistic diversity.How to Turn These Models into Your Own Free APIYou don't have to build an API framework from scratch. The developer community has created drop-in API wrappers that make these open-source models behave exactly like a paid cloud service.Here are the best tools to deploy your own zero-cost API:faster-whisper-server (Highly Recommended): This open-source project wraps an optimized version of Whisper inside a web server. It creates a local API endpoint (http://localhost:8000/v1/audio/transcriptions) that mimics OpenAI's official API layout. You can switch your app from OpenAI's paid tier to your own free server by changing just one line of code.  Whisper.cpp (Server Mode): If you are running on a Mac (Apple Silicon) or standard CPU hardware without a massive dedicated graphics card, Whisper.cpp is a C/C++ port optimized for pure speed. It features a built-in server tool to spin up a local API endpoint immediately.  Hugging Face Transformers: Both Whisper and Meta MMS are hosted for free on Hugging Face. You can use Python with the transformers library and FastAPI to create a custom API endpoint in less than 20 lines of code.Summary  To get 100% free, unlimited speech-to-text for 70+ languages, download faster-whisper-server and run the Whisper Large-v3 Turbo model locally. Your only cost will be the electricity to power your machine.
