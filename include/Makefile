sleef.h :
	+"$(MAKE)" --directory=../src/libm sleef.h

.PHONY: install
install : sleef.h sleefdft.h
	install -D -t $(DESTDIR)$(PREFIX)/include sleef.h sleefdft.h

.PHONY: uninstall
uninstall :
	rm -f $(DESTDIR)$(PREFIX)/include/sleef.h
	rm -f $(DESTDIR)$(PREFIX)/include/sleefdft.h

.PHONY: clean
clean :
	rm -f *~

.PHONY: distclean
distclean : clean
	rm -f sleef.h
