class ReviewlizeController < ApplicationController
  def home
  end

  def index
    if params[:search_word].match(/(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})/)
      if params[:search_word].match(/(?:https?:\/\/(?:www\.){0,1}amazon\.com(?:\/.*){0,1}(?:\/dp\/|\/gp\/product\/))(.+?)(?:\/.*|$)/)
        @results = AmazonProductDataScrapper.scrape(params[:search_word])
      else
        flash[:error] = "Invalid Url!"
        redirect_to root_path
      end
    else
      search_word = params[:search_word]
      @results = AmazonSearchScrapper.scrape(search_word)
      if @results.size == 0
        flash[:error] = "No Results found"
        redirect_to root_path
      end
    end
  end

  def scrape_product
    product_url = params[:product_url]
    @reviews = AmazonProductScrapper.scrape(product_url)

    response = Faraday.post("http://localhost:5000/predict", {sentences: @reviews}.to_json, "Content-Type" => "application/json")
    response = JSON.parse(response.body)
    #response = JSON.parse("{\"results\":{\"aspect_analize\":{\"funny\":{\"negative\":#{rand(10)},\"neutral\":0,\"positive\":#{rand(10)}},\"screen\":{\"negative\":#{rand(10)},\"neutral\":#{rand(10)},\"positive\":#{rand(10)}},\"signs\":{\"negative\":#{rand(10)},\"neutral\":3,\"positive\":#{rand(10)}},\"that\":{\"negative\":#{rand(10)},\"neutral\":1,\"positive\":#{rand(10)}},\"when\":{\"negative\":#{rand(10)},\"neutral\":1,\"positive\":#{rand(10)}}},\"aspects\":[\"that\",\"signs\",\"when\",\"signs\",\"funny\",\"screen\"],\"filtered\":[[\"signs\",\"minimal\"],[\"signs\",\"when\"],[\"signs\",\"wear\"],[\"signs\",\"damage\"],[\"signs\",\"showed\"],[\"screen\",\"great\"],[\"funny\",\"kind\"],[\"when\",\"notes\"],[\"that\",\"do\"]]},\"resultss\":1}")

    all_pos = 0
    all_neg = 0
    response["results"]["aspect_analize"].each do |k,v| 
      all_neg += v["negative"]
      all_pos += v["positive"]
    end
    rate = (all_pos.to_f/(all_pos+all_neg)*5).round(2)

    Product.find_by(url: product_url).update(result: response.to_json, rate: rate, analyzed_reviews_count: @reviews.size)

    response["rate"] = rate
    response["reviews_count"] = @reviews.size
    
    render json: response

  end

  def one_product_analysis
    @products = params[:products]
  end

  def analyze
    if current_user && params[:history]
      @products = []
      @history_record = HistoryRecord.find_by(id: params[:history_record_id])
      @history_record.history_products.each do |hp|
        @products << hp.product
      end
    else
      unless params[:products].present?
        flash[:error] = "Choose at least one product to analyze!"
        return redirect_back fallback_location: root_path
      end
      if current_user
        @history_record = current_user.history_records.create(search_title: params[:search_title])
      end
      @products = []
      params[:products].each do |prod|
        prod = eval(prod)
        product = Product.find_by(url: prod[:url])
        unless product.present?
          product = Product.create(name: prod[:title], url: prod[:url], image_url: prod[:image], price: prod[:price], supported_website: SupportedWebsite.find_by(base_url: "#{prod[:url].split('.com').first}.com"))
        end
        @products << product
        @history_record.history_products.create(product: product) if @history_record.present?
        @not_all_stored = true if product.result == nil
      end
      if @history_record.present?
        if @products.size > 1
          @history_record.update(analysis_type: 0)
        else
          @history_record.update(analysis_type: 1)
        end 
      end
    end
  end


end