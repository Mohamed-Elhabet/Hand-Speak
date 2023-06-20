const carousel = document.querySelector('.carousel');
const slides = carousel.querySelector('.slides');
const slideWidth = carousel.clientWidth;
let currentIndex = 0;

const goToSlide = (index) => {
  slides.style.transform = `translateX(-${index * slideWidth}px)`;
};

const nextSlide = () => {
  currentIndex = (currentIndex + 1) % slides.children.length;
  goToSlide(currentIndex);
};

const prevSlide = () => {
  currentIndex = (currentIndex - 1 + slides.children.length) % slides.children.length;
  goToSlide(currentIndex);
};

carousel.querySelector('.btn--left').addEventListener('click', prevSlide);
carousel.querySelector('.btn--right').addEventListener('click', nextSlide);

goToSlide(currentIndex);
