Rails.application.routes.draw do
  devise_for :users
  # For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html
  root to: 'reviewlize#home'
  get 'search', to: "reviewlize#index", as: :search
  get 'scrape_product', to: "reviewlize#scrape_product", as: :scrape_product
  get "analyze", to: 'reviewlize#analyze', as: :analyze_products

  get "history", to: 'users#history', as: :user_history
end
